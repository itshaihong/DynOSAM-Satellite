#include "dynosam/dataprovider/KubricDataProvider.hpp"

#include <glog/logging.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/pipeline/ThreadSafeTemporalBuffer.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {

// ============================================================================
// Hardcoded Kubric camera intrinsics
//
// Source: Kubric JSON metadata, NDC K matrix converted to pixel space.
//
//   fx = K_ndc[0,0] * W = 1.557318330489192 * 1920 = 2990.051 px
//   fy = |K_ndc[1,1]| * H = 2.491709328782707 * 1200 = 2990.051 px
//   cx = W / 2 = 960 px
//   cy = H / 2 = 600 px
//
// Cross-check: 2 * atan(960 / 2990.051) = 35.60 deg  == fov_horizontal_deg
// ============================================================================
static constexpr double kFx     = 2990.051195;
static constexpr double kFy     = 2990.051195;
static constexpr double kCx     = 960.0;
static constexpr double kCy     = 600.0;
static constexpr int    kWidth  = 1920;
static constexpr int    kHeight = 1200;

// ============================================================================
// KubricAllLoader
//   Owns all disk I/O and ground-truth assembly.  Mirrors OMDOldAllLoader
//   but is adapted for the Kubric dataset layout:
//
//   Spacecraft/
//   ├── image/                   .png  RGB
//   ├── depth/                   .tiff metric depth (sentinel 9999 m)
//   ├── flow/                    .flo  forward optical flow
//   ├── seg/                     .png  binary mask {0, 1}
//   ├── times.txt                one timestamp per line (same as OMD)
//   ├── pose_ground_truth.txt    per-line: timestamp tx ty tz qx qy qz qw
//   └── object_pose_ground_truth.txt per-line: timestamp tx ty tz qx qy qz qw
// ============================================================================
class KubricAllLoader {
 public:
  DYNO_POINTER_TYPEDEFS(KubricAllLoader)

  explicit KubricAllLoader(const std::string& dataset_path) {
    loadAll(dataset_path);
  }

  // ── size / accessors ──────────────────────────────────────────────────────

  size_t size() const { return rgb_file_names_.size(); }

  double getTimestamp(size_t idx) const {
    return static_cast<double>(timestamps_.at(idx));
  }

  // RGB: load .png as BGR then convert to RGB (matches OMD behaviour)
  cv::Mat getRGB(size_t idx) const {
    CHECK_LT(idx, rgb_file_names_.size());
    cv::Mat rgb;
    utils::loadRGB(rgb_file_names_.at(idx), rgb);
    CHECK(!rgb.empty()) << "Empty RGB at " << rgb_file_names_.at(idx);
    img_size_ = rgb.size();
    return rgb;
  }

  // Depth: float32 TIFF in metres, converted to CV_64F.
  // Pixels with value >= 9998 m are Kubric's "no-hit" sentinel and are
  // zeroed so DynoSAM treats them as invalid (same convention as OMD's
  // zero-disparity pixels).
  cv::Mat getDepthImage(size_t idx) const {
      CHECK_LT(idx, depth_file_names_.size());

      cv::Mat depth;
      utils::loadDepth(depth_file_names_.at(idx), depth);
      LOG(INFO) << "[Kubric] depth type=" << depth.type()
          << " channels=" << depth.channels()
          << " depth=" << depth.depth();
      CHECK(!depth.empty()) << "Empty depth image at "
                            << depth_file_names_.at(idx);

      // Kubric uses 9999.0 m as a sentinel for rays that hit nothing.
      // Zero these out so DynoSAM treats them as invalid depth,
      // consistent with the zero-disparity convention in OMD.
      // depth is CV_64F after loadDepth, so threshold operates correctly.
      cv::Mat invalid_mask;
      cv::threshold(depth, invalid_mask, 9998.0, 1.0, cv::THRESH_BINARY);
      invalid_mask.convertTo(invalid_mask, CV_64F);
      depth.setTo(0.0, invalid_mask > 0.5);

      return depth;
  }

  // Optical flow: .flo binary format, loaded via DynoSAM utility
  cv::Mat getOpticalFlow(size_t idx) const {
    CHECK_LT(idx, flow_file_names_.size());
    cv::Mat flow;
    utils::loadFlow(flow_file_names_.at(idx), flow);
    CHECK(!flow.empty()) << "Empty flow at " << flow_file_names_.at(idx);
    return flow;
  }

  cv::Mat getInstanceMask(size_t idx) const {
      CHECK_LT(idx, seg_file_names_.size());

      cv::Mat mask;
      utils::loadMask(seg_file_names_.at(idx), mask);
      CHECK(!mask.empty()) << "Empty segmentation mask at "
                          << seg_file_names_.at(idx);
      return mask;
  }

  GroundTruthInputPacket getGtPacket(size_t idx) const {
    return ground_truths_.at(idx);
  }

  const CameraParams& getCameraParams() const { return camera_params_; }

  // ── loadAll ───────────────────────────────────────────────────────────────

  void loadAll(const std::string& path) {
    loadTimestamps(path);
    loadFileLists(path);
    loadCameraPoses(path);
    loadObjectPoses(path);
    assembleGroundTruth();
  }

 private:
  // ── timestamp loading (identical to OMD) ──────────────────────────────────

  void loadTimestamps(const std::string& path) {
    const std::string times_file = path + "/times.txt";
    utils::throwExceptionIfPathInvalid(times_file);

    std::ifstream stream(times_file);
    std::string line;
    while (std::getline(stream, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      double t;
      ss >> t;
      timestamps_.push_back(t);
    }
    LOG(INFO) << "[Kubric] Loaded " << timestamps_.size() << " timestamps";
  }

  // ── file list loading ─────────────────────────────────────────────────────

  void loadFileLists(const std::string& path) {
    utils::loadPathsInDirectory(
        rgb_file_names_, path + "/image/",
        [](const std::string& f) {
          return boost::algorithm::ends_with(f, ".png");
        });

    utils::loadPathsInDirectory(
        depth_file_names_, path + "/depth/",
        [](const std::string& f) {
          return boost::algorithm::ends_with(f, ".tiff") ||
                 boost::algorithm::ends_with(f, ".tif");
        });

    utils::loadPathsInDirectory(
        flow_file_names_, path + "/flow/",
        [](const std::string& f) {
          return boost::algorithm::ends_with(f, ".flo");
        });

    utils::loadPathsInDirectory(
        seg_file_names_, path + "/seg/",
        [](const std::string& f) {
          return boost::algorithm::ends_with(f, ".png");
        });

    const size_t N = rgb_file_names_.size();
    LOG(INFO) << "[Kubric] RGB=" << N
              << " depth=" << depth_file_names_.size()
              << " flow="  << flow_file_names_.size()
              << " seg="   << seg_file_names_.size();

    CHECK_EQ(depth_file_names_.size(), N)
        << "Depth file count must match RGB count";
    CHECK_EQ(seg_file_names_.size(), N)
        << "Seg file count must match RGB count";
    CHECK_GE(flow_file_names_.size(), N - 1)
        << "Need at least N-1 flow files for N frames";
    CHECK_GE(timestamps_.size(), N)
        << "Need at least as many timestamps as frames";
  }

  // ── camera pose loading ───────────────────────────────────────────────────
  // Format (one row per frame, no header):
  //   timestamp  tx  ty  tz  qx  qy  qz  qw
  //
  // Quaternion convention: qx qy qz qw  (ROS / Eigen)
  // GTSAM Rot3::Quaternion expects (w, x, y, z)

  void loadCameraPoses(const std::string& path) {
    const std::string pose_file = path + "/pose_ground_truth.txt";
    utils::throwExceptionIfPathInvalid(pose_file);

    std::ifstream stream(pose_file);
    std::string line;
    while (std::getline(stream, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);

      double ts, tx, ty, tz, qx, qy, qz, qw;
      ss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

      gtsam::Rot3    rot = gtsam::Rot3::Quaternion(qw, qx, qy, qz);
      gtsam::Point3  t(tx, ty, tz);
      vPoseGT_.emplace_back(rot, t);
    }
    LOG(INFO) << "[Kubric] Loaded " << vPoseGT_.size() << " camera poses";
  }

  // ── object pose loading ───────────────────────────────────────────────────
  // Format (one row per object per frame, no header):
  //   timestamp  tx  ty  tz  qx  qy  qz  qw
  //
  // Object ID is assigned by appearance order within each frame (0-based).
  // If your file gains an explicit object-ID column, replace the
  // objects_per_frame counter with a direct parse.

  void loadObjectPoses(const std::string& path) {
    const std::string obj_file = path + "/object_pose_ground_truth.txt";
    utils::throwExceptionIfPathInvalid(obj_file);

    // Map timestamp → frame index for matching
    std::unordered_map<double, size_t> ts_to_frame;
    for (size_t i = 0; i < timestamps_.size(); ++i)
      ts_to_frame[timestamps_[i]] = i;

    // Track how many objects we have seen per frame to assign IDs
    std::unordered_map<size_t, size_t> objects_per_frame;

    std::ifstream stream(obj_file);
    std::string line;
    while (std::getline(stream, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);

      double ts, tx, ty, tz, qx, qy, qz, qw;
      ss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

      // Match to frame index via timestamp
      auto it = ts_to_frame.find(ts);
      if (it == ts_to_frame.end()) {
        LOG(WARNING) << "[Kubric] Object pose timestamp " << ts
                     << " not found in times.txt — skipping";
        continue;
      }
      const size_t frame_id = it->second;

      ObjectPoseGT obj;
      obj.frame_id_  = frame_id;
      obj.object_id_ = objects_per_frame[frame_id]++;

      gtsam::Rot3   rot = gtsam::Rot3::Quaternion(qw, qx, qy, qz);
      gtsam::Point3 t(tx, ty, tz);
      obj.L_world_ = gtsam::Pose3(rot, t);

      vObjPoseGT_.push_back(obj);
    }
    LOG(INFO) << "[Kubric] Loaded " << vObjPoseGT_.size() << " object poses";
  }

  // ── ground-truth packet assembly (mirrors OMD exactly) ───────────────────

  void assembleGroundTruth() {
    const size_t N = rgb_file_names_.size();

    // Build per-frame object-pose index
    std::vector<std::vector<size_t>> vObjPoseID(N);
    for (size_t i = 0; i < vObjPoseGT_.size(); ++i) {
      const size_t fid = vObjPoseGT_[i].frame_id_;
      if (fid < N) vObjPoseID[fid].push_back(i);
    }

    // Align all poses relative to the first camera frame
    CHECK(!vPoseGT_.empty()) << "No camera poses loaded";
    const gtsam::Pose3 initial_pose = vPoseGT_[0];

    for (size_t fid = 0; fid < N - 1; ++fid) {
      GroundTruthInputPacket gt;
      gt.timestamp_ = timestamps_[fid];
      gt.frame_id_  = fid;

      const gtsam::Pose3& original_camera_pose = vPoseGT_[fid];
      const gtsam::Pose3  aligned_camera_pose  =
          initial_pose.inverse() * original_camera_pose;

      gt.X_world_ = aligned_camera_pose;

      for (size_t i = 0; i < vObjPoseID[fid].size(); ++i) {
        ObjectPoseGT obj = vObjPoseGT_[vObjPoseID[fid][i]];

        const gtsam::Pose3 relative_object_pose =
            original_camera_pose.inverse() * obj.L_world_;
        const gtsam::Pose3 aligned_object_pose =
            aligned_camera_pose * relative_object_pose;

        obj.L_camera_ = relative_object_pose;
        obj.L_world_  = aligned_object_pose;

        gt.object_poses_.push_back(obj);
        CHECK_EQ(gt.object_poses_[i].frame_id_, fid);
      }

      if (fid > 0) {
        auto& prev = ground_truths_.at(fid - 1);
        gt.calculateAndSetMotions(prev);
      }

      ground_truths_.push_back(gt);
    }

    LOG(INFO) << "[Kubric] Assembled " << ground_truths_.size()
              << " GT packets";

    // Build hardcoded camera params
    CameraParams::IntrinsicsCoeffs intrinsics = {kFx, kFy, kCx, kCy};
    CameraParams::DistortionCoeffs distortion = {0.0, 0.0, 0.0, 0.0};
    cv::Size image_size(kWidth, kHeight);
    auto model = CameraParams::stringToDistortion("radtan", "pinhole");
    camera_params_ = CameraParams(intrinsics, distortion, image_size, model);

    LOG(INFO) << "[Kubric] Camera params: " << camera_params_.toString();
  }

  // ── data members ──────────────────────────────────────────────────────────

  std::vector<std::string>             rgb_file_names_;
  std::vector<std::string>             depth_file_names_;
  std::vector<std::string>             flow_file_names_;
  std::vector<std::string>             seg_file_names_;
  std::vector<double>                  timestamps_;

  std::vector<gtsam::Pose3>            vPoseGT_;
  std::vector<ObjectPoseGT>            vObjPoseGT_;
  std::vector<GroundTruthInputPacket>  ground_truths_;

  mutable cv::Size img_size_;
  CameraParams     camera_params_;
};

// ============================================================================
// Thin TimestampBaseLoader adapter (same pattern as OMMTimestampLoader)
// ============================================================================
struct KubricTimestampLoader : public TimestampBaseLoader {
  KubricAllLoader::Ptr loader_;

  explicit KubricTimestampLoader(KubricAllLoader::Ptr loader)
      : loader_(CHECK_NOTNULL(loader)) {}

  std::string getFolderName() const override { return ""; }
  size_t      size()          const override { return loader_->size(); }
  double      getItem(size_t idx)    override { return loader_->getTimestamp(idx); }
};

// ============================================================================
// KubricDataLoader — public entry point
// ============================================================================
KubricDataLoader::KubricDataLoader(const fs::path& dataset_path)
    : KubricDatasetProvider(dataset_path) {
  LOG(INFO) << "[Kubric] Starting KubricDataLoader with path " << dataset_path;

  auto loader = std::make_shared<KubricAllLoader>(dataset_path.string());

  left_camera_params_ = loader->getCameraParams();
  CHECK(getCameraParams());

  auto timestamp_loader =
      std::make_shared<KubricTimestampLoader>(loader);

  auto rgb_loader =
      std::make_shared<FunctionalDataFolder<cv::Mat>>(
          [loader](size_t idx) { return loader->getRGB(idx); });

  auto optical_flow_loader =
      std::make_shared<FunctionalDataFolder<cv::Mat>>(
          [loader](size_t idx) { return loader->getOpticalFlow(idx); });

  auto depth_loader =
      std::make_shared<FunctionalDataFolder<cv::Mat>>(
          [loader](size_t idx) { return loader->getDepthImage(idx); });

  auto instance_mask_loader =
      std::make_shared<FunctionalDataFolder<cv::Mat>>(
          [loader](size_t idx) { return loader->getInstanceMask(idx); });

  auto gt_loader =
      std::make_shared<FunctionalDataFolder<GroundTruthInputPacket>>(
          [loader](size_t idx) { return loader->getGtPacket(idx); });

  this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader,
                   depth_loader, instance_mask_loader, gt_loader);

  auto callback = [&](size_t frame_id, Timestamp timestamp,
                      cv::Mat rgb, cv::Mat optical_flow,
                      cv::Mat depth, cv::Mat instance_mask,
                      GroundTruthInputPacket gt_packet) -> bool {
    CHECK_EQ(timestamp, gt_packet.timestamp_);

    if (ground_truth_packet_callback_)
      ground_truth_packet_callback_(gt_packet);

    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
                   .depth(depth)
                   .opticalFlow(optical_flow)
                   .objectMotionMask(instance_mask);

    if (image_container_callback_)
      image_container_callback_(
          std::make_shared<ImageContainer>(image_container));

    return true;
  };

  this->setCallback(callback);
}

}  // namespace dyno
