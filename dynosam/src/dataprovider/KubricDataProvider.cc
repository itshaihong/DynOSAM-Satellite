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

// namespace dyno {

// // ============================================================================
// // Hardcoded Kubric camera intrinsics
// //
// // Source: Kubric JSON metadata, NDC K matrix converted to pixel space.
// //
// //   fx = K_ndc[0,0] * W = 1.557318330489192 * 1920 = 2990.051 px
// //   fy = |K_ndc[1,1]| * H = 2.491709328782707 * 1200 = 2990.051 px
// //   cx = W / 2 = 960 px
// //   cy = H / 2 = 600 px
// //
// // Cross-check: 2 * atan(960 / 2990.051) = 35.60 deg  == fov_horizontal_deg
// // ============================================================================
// static constexpr double kFx     = 2990.051195;
// static constexpr double kFy     = 2990.051195;
// static constexpr double kCx     = 960.0;
// static constexpr double kCy     = 600.0;
// static constexpr int    kWidth  = 1920;
// static constexpr int    kHeight = 1200;

// // ============================================================================
// // KubricAllLoader
// //   Owns all disk I/O and ground-truth assembly.  Mirrors OMDOldAllLoader
// //   but is adapted for the Kubric dataset layout:
// //
// //   Spacecraft/
// //   ├── image/                   .png  RGB
// //   ├── depth/                   .tiff metric depth (sentinel 9999 m)
// //   ├── flow/                    .flo  forward optical flow
// //   ├── seg/                     .png  binary mask {0, 1}
// //   ├── times.txt                one timestamp per line (same as OMD)
// //   ├── pose_ground_truth.txt    per-line: timestamp tx ty tz qx qy qz qw
// //   └── object_pose_ground_truth.txt per-line: timestamp tx ty tz qx qy qz qw
// // ============================================================================
// class KubricAllLoader {
//  public:
//   DYNO_POINTER_TYPEDEFS(KubricAllLoader)

//   explicit KubricAllLoader(const std::string& dataset_path) {
//     loadAll(dataset_path);
//   }

//   // ── size / accessors ──────────────────────────────────────────────────────

//   size_t size() const { return rgb_file_names_.size(); }

//   double getTimestamp(size_t idx) const {
//     return static_cast<double>(timestamps_.at(idx));
//   }

//   // RGB: load .png as BGR then convert to RGB (matches OMD behaviour)
//   cv::Mat getRGB(size_t idx) const {
//     CHECK_LT(idx, rgb_file_names_.size());
//     cv::Mat rgb;
//     utils::loadRGB(rgb_file_names_.at(idx), rgb);
//     CHECK(!rgb.empty()) << "Empty RGB at " << rgb_file_names_.at(idx);
//     img_size_ = rgb.size();
//     return rgb;
//   }

//   // Depth: float32 TIFF in metres, converted to CV_64F.
//   // Pixels with value >= 9998 m are Kubric's "no-hit" sentinel and are
//   // zeroed so DynoSAM treats them as invalid (same convention as OMD's
//   // zero-disparity pixels).
//   cv::Mat getDepthImage(size_t idx) const {
//       CHECK_LT(idx, depth_file_names_.size());

//       cv::Mat depth;
//       utils::loadDepth(depth_file_names_.at(idx), depth);
//       LOG(INFO) << "[Kubric] depth type=" << depth.type()
//           << " channels=" << depth.channels()
//           << " depth=" << depth.depth();
//       CHECK(!depth.empty()) << "Empty depth image at "
//                             << depth_file_names_.at(idx);

//       // Kubric uses 9999.0 m as a sentinel for rays that hit nothing.
//       // Zero these out so DynoSAM treats them as invalid depth,
//       // consistent with the zero-disparity convention in OMD.
//       // depth is CV_64F after loadDepth, so threshold operates correctly.
//       cv::Mat invalid_mask;
//       cv::threshold(depth, invalid_mask, 9998.0, 1.0, cv::THRESH_BINARY);
//       invalid_mask.convertTo(invalid_mask, CV_64F);
//       depth.setTo(0.0, invalid_mask > 0.5);

//       return depth;
//   }

//   // Optical flow: .flo binary format, loaded via DynoSAM utility
//   cv::Mat getOpticalFlow(size_t idx) const {
//     CHECK_LT(idx, flow_file_names_.size());
//     cv::Mat flow;
//     utils::loadFlow(flow_file_names_.at(idx), flow);
//     CHECK(!flow.empty()) << "Empty flow at " << flow_file_names_.at(idx);
//     return flow;
//   }

//   cv::Mat getInstanceMask(size_t idx) const {
//       CHECK_LT(idx, seg_file_names_.size());

//       cv::Mat mask;
//       utils::loadMask(seg_file_names_.at(idx), mask);
//       CHECK(!mask.empty()) << "Empty segmentation mask at "
//                           << seg_file_names_.at(idx);
//       return mask;
//   }

//   GroundTruthInputPacket getGtPacket(size_t idx) const {
//     return ground_truths_.at(idx);
//   }

//   const CameraParams& getCameraParams() const { return camera_params_; }

//   // ── loadAll ───────────────────────────────────────────────────────────────

//   void loadAll(const std::string& path) {
//     loadTimestamps(path);
//     loadFileLists(path);
//     loadCameraPoses(path);
//     loadObjectPoses(path);
//     assembleGroundTruth();
//   }

//  private:
//   // ── timestamp loading (identical to OMD) ──────────────────────────────────

//   void loadTimestamps(const std::string& path) {
//     const std::string times_file = path + "/times.txt";
//     utils::throwExceptionIfPathInvalid(times_file);

//     std::ifstream stream(times_file);
//     std::string line;
//     while (std::getline(stream, line)) {
//       if (line.empty() || line[0] == '#') continue;
//       std::istringstream ss(line);
//       double t;
//       ss >> t;
//       timestamps_.push_back(t);
//     }
//     LOG(INFO) << "[Kubric] Loaded " << timestamps_.size() << " timestamps";
//   }

//   // ── file list loading ─────────────────────────────────────────────────────

//   void loadFileLists(const std::string& path) {
//     utils::loadPathsInDirectory(
//         rgb_file_names_, path + "/image/",
//         [](const std::string& f) {
//           return boost::algorithm::ends_with(f, ".png");
//         });

//     utils::loadPathsInDirectory(
//         depth_file_names_, path + "/depth/",
//         [](const std::string& f) {
//           return boost::algorithm::ends_with(f, ".tiff") ||
//                  boost::algorithm::ends_with(f, ".tif");
//         });

//     utils::loadPathsInDirectory(
//         flow_file_names_, path + "/flow/",
//         [](const std::string& f) {
//           return boost::algorithm::ends_with(f, ".flo");
//         });

//     utils::loadPathsInDirectory(
//         seg_file_names_, path + "/seg/",
//         [](const std::string& f) {
//           return boost::algorithm::ends_with(f, ".png");
//         });

//     const size_t N = rgb_file_names_.size();
//     LOG(INFO) << "[Kubric] RGB=" << N
//               << " depth=" << depth_file_names_.size()
//               << " flow="  << flow_file_names_.size()
//               << " seg="   << seg_file_names_.size();

//     CHECK_EQ(depth_file_names_.size(), N)
//         << "Depth file count must match RGB count";
//     CHECK_EQ(seg_file_names_.size(), N)
//         << "Seg file count must match RGB count";
//     CHECK_GE(flow_file_names_.size(), N - 1)
//         << "Need at least N-1 flow files for N frames";
//     CHECK_GE(timestamps_.size(), N)
//         << "Need at least as many timestamps as frames";
//   }

//   // ── camera pose loading ───────────────────────────────────────────────────
//   // Format (one row per frame, no header):
//   //   timestamp  tx  ty  tz  qx  qy  qz  qw
//   //
//   // Quaternion convention: qx qy qz qw  (ROS / Eigen)
//   // GTSAM Rot3::Quaternion expects (w, x, y, z)

//   void loadCameraPoses(const std::string& path) {
//     const std::string pose_file = path + "/pose_ground_truth.txt";
//     utils::throwExceptionIfPathInvalid(pose_file);

//     std::ifstream stream(pose_file);
//     std::string line;
//     while (std::getline(stream, line)) {
//       if (line.empty() || line[0] == '#') continue;
//       std::istringstream ss(line);

//       double ts, tx, ty, tz, qx, qy, qz, qw;
//       ss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

//       gtsam::Rot3    rot = gtsam::Rot3::Quaternion(qw, qx, qy, qz);
//       gtsam::Point3  t(tx, ty, tz);
//       vPoseGT_.emplace_back(rot, t);
//     }
//     LOG(INFO) << "[Kubric] Loaded " << vPoseGT_.size() << " camera poses";
//   }

//   // ── object pose loading ───────────────────────────────────────────────────
//   // Format (one row per object per frame, no header):
//   //   timestamp  tx  ty  tz  qx  qy  qz  qw
//   //
//   // Object ID is assigned by appearance order within each frame (0-based).
//   // If your file gains an explicit object-ID column, replace the
//   // objects_per_frame counter with a direct parse.

//   void loadObjectPoses(const std::string& path) {
//     const std::string obj_file = path + "/object_pose_ground_truth.txt";
//     utils::throwExceptionIfPathInvalid(obj_file);

//     // Map timestamp → frame index for matching
//     std::unordered_map<double, size_t> ts_to_frame;
//     for (size_t i = 0; i < timestamps_.size(); ++i)
//       ts_to_frame[timestamps_[i]] = i;

//     // Track how many objects we have seen per frame to assign IDs
//     std::unordered_map<size_t, size_t> objects_per_frame;

//     std::ifstream stream(obj_file);
//     std::string line;
//     while (std::getline(stream, line)) {
//       if (line.empty() || line[0] == '#') continue;
//       std::istringstream ss(line);

//       double ts, tx, ty, tz, qx, qy, qz, qw;
//       ss >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

//       // Match to frame index via timestamp
//       auto it = ts_to_frame.find(ts);
//       if (it == ts_to_frame.end()) {
//         LOG(WARNING) << "[Kubric] Object pose timestamp " << ts
//                      << " not found in times.txt — skipping";
//         continue;
//       }
//       const size_t frame_id = it->second;

//       ObjectPoseGT obj;
//       obj.frame_id_  = frame_id;
//       obj.object_id_ = objects_per_frame[frame_id]++;

//       gtsam::Rot3   rot = gtsam::Rot3::Quaternion(qw, qx, qy, qz);
//       gtsam::Point3 t(tx, ty, tz);
//       obj.L_world_ = gtsam::Pose3(rot, t);

//       vObjPoseGT_.push_back(obj);
//     }
//     LOG(INFO) << "[Kubric] Loaded " << vObjPoseGT_.size() << " object poses";
//   }

//   // ── ground-truth packet assembly (mirrors OMD exactly) ───────────────────

//   void assembleGroundTruth() {
//     const size_t N = rgb_file_names_.size();

//     // Build per-frame object-pose index
//     std::vector<std::vector<size_t>> vObjPoseID(N);
//     for (size_t i = 0; i < vObjPoseGT_.size(); ++i) {
//       const size_t fid = vObjPoseGT_[i].frame_id_;
//       if (fid < N) vObjPoseID[fid].push_back(i);
//     }

//     // Align all poses relative to the first camera frame
//     CHECK(!vPoseGT_.empty()) << "No camera poses loaded";
//     const gtsam::Pose3 initial_pose = vPoseGT_[0];

//     for (size_t fid = 0; fid < N - 1; ++fid) {
//       GroundTruthInputPacket gt;
//       gt.timestamp_ = timestamps_[fid];
//       gt.frame_id_  = fid;

//       const gtsam::Pose3& original_camera_pose = vPoseGT_[fid];
//       const gtsam::Pose3  aligned_camera_pose  =
//           initial_pose.inverse() * original_camera_pose;

//       gt.X_world_ = aligned_camera_pose;

//       for (size_t i = 0; i < vObjPoseID[fid].size(); ++i) {
//         ObjectPoseGT obj = vObjPoseGT_[vObjPoseID[fid][i]];

//         const gtsam::Pose3 relative_object_pose =
//             original_camera_pose.inverse() * obj.L_world_;
//         const gtsam::Pose3 aligned_object_pose =
//             aligned_camera_pose * relative_object_pose;

//         obj.L_camera_ = relative_object_pose;
//         obj.L_world_  = aligned_object_pose;

//         gt.object_poses_.push_back(obj);
//         CHECK_EQ(gt.object_poses_[i].frame_id_, fid);
//       }

//       if (fid > 0) {
//         auto& prev = ground_truths_.at(fid - 1);
//         gt.calculateAndSetMotions(prev);
//       }

//       ground_truths_.push_back(gt);
//     }

//     LOG(INFO) << "[Kubric] Assembled " << ground_truths_.size()
//               << " GT packets";

//     // Build hardcoded camera params
//     CameraParams::IntrinsicsCoeffs intrinsics = {kFx, kFy, kCx, kCy};
//     CameraParams::DistortionCoeffs distortion = {0.0, 0.0, 0.0, 0.0};
//     cv::Size image_size(kWidth, kHeight);
//     auto model = CameraParams::stringToDistortion("radtan", "pinhole");
//     camera_params_ = CameraParams(intrinsics, distortion, image_size, model);

//     LOG(INFO) << "[Kubric] Camera params: " << camera_params_.toString();
//   }

//   // ── data members ──────────────────────────────────────────────────────────

//   std::vector<std::string>             rgb_file_names_;
//   std::vector<std::string>             depth_file_names_;
//   std::vector<std::string>             flow_file_names_;
//   std::vector<std::string>             seg_file_names_;
//   std::vector<double>                  timestamps_;

//   std::vector<gtsam::Pose3>            vPoseGT_;
//   std::vector<ObjectPoseGT>            vObjPoseGT_;
//   std::vector<GroundTruthInputPacket>  ground_truths_;

//   mutable cv::Size img_size_;
//   CameraParams     camera_params_;
// };

// // ============================================================================
// // Thin TimestampBaseLoader adapter (same pattern as OMMTimestampLoader)
// // ============================================================================
// struct KubricTimestampLoader : public TimestampBaseLoader {
//   KubricAllLoader::Ptr loader_;

//   explicit KubricTimestampLoader(KubricAllLoader::Ptr loader)
//       : loader_(CHECK_NOTNULL(loader)) {}

//   std::string getFolderName() const override { return ""; }
//   size_t      size()          const override { return loader_->size(); }
//   double      getItem(size_t idx)    override { return loader_->getTimestamp(idx); }
// };

// // ============================================================================
// // KubricDataLoader — public entry point
// // ============================================================================
// KubricDataLoader::KubricDataLoader(const fs::path& dataset_path)
//     : KubricDatasetProvider(dataset_path) {
//   LOG(INFO) << "[Kubric] Starting KubricDataLoader with path " << dataset_path;

//   auto loader = std::make_shared<KubricAllLoader>(dataset_path.string());

//   left_camera_params_ = loader->getCameraParams();
//   CHECK(getCameraParams());

//   auto timestamp_loader =
//       std::make_shared<KubricTimestampLoader>(loader);

//   auto rgb_loader =
//       std::make_shared<FunctionalDataFolder<cv::Mat>>(
//           [loader](size_t idx) { return loader->getRGB(idx); });

//   auto optical_flow_loader =
//       std::make_shared<FunctionalDataFolder<cv::Mat>>(
//           [loader](size_t idx) { return loader->getOpticalFlow(idx); });

//   auto depth_loader =
//       std::make_shared<FunctionalDataFolder<cv::Mat>>(
//           [loader](size_t idx) { return loader->getDepthImage(idx); });

//   auto instance_mask_loader =
//       std::make_shared<FunctionalDataFolder<cv::Mat>>(
//           [loader](size_t idx) { return loader->getInstanceMask(idx); });

//   auto gt_loader =
//       std::make_shared<FunctionalDataFolder<GroundTruthInputPacket>>(
//           [loader](size_t idx) { return loader->getGtPacket(idx); });

//   this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader,
//                    depth_loader, instance_mask_loader, gt_loader);

//   auto callback = [&](size_t frame_id, Timestamp timestamp,
//                       cv::Mat rgb, cv::Mat optical_flow,
//                       cv::Mat depth, cv::Mat instance_mask,
//                       GroundTruthInputPacket gt_packet) -> bool {
//     CHECK_EQ(timestamp, gt_packet.timestamp_);

//     if (ground_truth_packet_callback_)
//       ground_truth_packet_callback_(gt_packet);

//     ImageContainer image_container(frame_id, timestamp);
//     image_container.rgb(rgb)
//                    .depth(depth)
//                    .opticalFlow(optical_flow)
//                    .objectMotionMask(instance_mask);

//     if (image_container_callback_)
//       image_container_callback_(
//           std::make_shared<ImageContainer>(image_container));

//     return true;
//   };

//   this->setCallback(callback);
// }

// }  // namespace dyno



namespace dyno {

class KubricAllLoader {
public:
  DYNO_POINTER_TYPEDEFS(KubricAllLoader)

  KubricAllLoader(const std::string& file_path) {
    // 1. Setup paths based on Space Sense folder structure
    const auto flow_image_path = file_path + "/flow/";
    utils::throwExceptionIfPathInvalid(flow_image_path);
    loadFlowImagesAndSize(flow_0_paths_, dataset_size_, flow_image_path);

    const auto rgb_image_path = file_path + "/image/";
    const auto depth_image_path = file_path + "/depth/";
    const auto mask_image_path = file_path + "/seg/";

    utils::throwExceptionIfPathInvalid(rgb_image_path);
    utils::throwExceptionIfPathInvalid(depth_image_path);
    utils::throwExceptionIfPathInvalid(mask_image_path);

    // 2. Load file paths and determine dataset size from RGB images (since no flow)
    loadImagesAndSize(image_paths_, dataset_size_, rgb_image_path, ".png");
    loadImages(depth_paths_, depth_image_path, ".tiff");
    loadImages(mask_paths_, mask_image_path, ".png");

    CHECK_EQ(image_paths_.size(), depth_paths_.size()) << "Mismatch in RGB and Depth image count!";
    CHECK_EQ(image_paths_.size(), mask_paths_.size()) << "Mismatch in RGB and Mask image count!";

    // 3. Set Camera Intrinsics 
    // TODO: Replace with your actual Space Sense dataset camera intrinsics!
    camera_params_ = CameraParams(
        CameraParams::IntrinsicsCoeffs({960, 960, 600, 600}), 
        CameraParams::DistortionCoeffs({0, 0, 0, 0}),
        cv::Size(1920, 1200), // Replace with actual resolution
        DistortionModel::RADTAN
    );

    // camera_params_ = CameraParams(
    //     CameraParams::IntrinsicsCoeffs({512, 512, 256, 256}), 
    //     CameraParams::DistortionCoeffs({0, 0, 0, 0}),
    //     cv::Size(1024, 1024), // Replace with actual resolution
    //     DistortionModel::RADTAN
    // );

    // 4. Load Timestamps
    const auto times_file_path = file_path + "/times.txt";
    utils::throwExceptionIfPathInvalid(times_file_path);
    loadTimes(times_file_path);

    // 5. Load Ground Truth Poses
    const auto gt_file_path = file_path + "/pose_ground_truth.txt";
    utils::throwExceptionIfPathInvalid(gt_file_path);
    loadGroundTruth(gt_file_path);
  }

  cv::Mat getRGB(size_t idx) const {
    CHECK_LT(idx, image_paths_.size());
    cv::Mat rgb;
    utils::loadRGB(image_paths_.at(idx), rgb);
    CHECK(!rgb.empty()) << "Failed to load RGB image at " << image_paths_.at(idx);
    return rgb;
  }

  cv::Mat getOpticalFlow(size_t idx) const {
    if (idx == 0 || idx >= flow_0_paths_.size()) {
        // Return a zero-motion matrix for the boundary frame
        return cv::Mat::zeros(cv::Size(1920, 1200), CV_32FC2);
        // return cv::Mat::zeros(cv::Size(1024, 1024), CV_32FC2);
    }
    CHECK_LT(idx, flow_0_paths_.size());

    cv::Mat flow;
    utils::loadFlow(flow_0_paths_.at(idx), flow);
    CHECK(!flow.empty());
    return flow;
  }

  cv::Mat getDepthImage(size_t idx) const {
    CHECK_LT(idx, depth_paths_.size());
    // Use IMREAD_ANYDEPTH to preserve the float32 format from the TIFF file
    cv::Mat depth = cv::imread(depth_paths_.at(idx), cv::IMREAD_ANYDEPTH);
    CHECK(!depth.empty()) << "Failed to load TIFF depth image at " << depth_paths_.at(idx);
    
    // DynoSAM/GTSAM backend optimization prefers CV_64F (double)
    // TODO: check if double is still needed
    cv::Mat depth_double;
    depth.convertTo(depth_double, CV_64F);
    return depth_double;
  }

  cv::Mat getInstanceMask(size_t idx) const {
    CHECK_LT(idx, mask_paths_.size());
    // ImageContainer::objectMotionMask is an instance-label image.  It must
    // therefore be CV_32SC1: one signed integer object id per pixel.
    //
    // The Kubric PNGs are RGBA (even though their R/G/B label values are the
    // same).  convertTo changes only the depth, not the channel count, so the
    // old code returned CV_32SC4.  Downstream code accesses this image with
    // ptr<ObjectId>(), assuming one value per pixel; this corrupts labels and
    // eventually causes OpenCV's feature-sampling calls to fail.
    cv::Mat encoded_mask =
        cv::imread(mask_paths_.at(idx), cv::IMREAD_UNCHANGED);
    CHECK(!encoded_mask.empty()) << "Failed to load instance mask at "
                                 << mask_paths_.at(idx);

    // Keep the source precision (Kubric masks may be 8- or 16-bit), but
    // explicitly discard the duplicated colour/alpha channels.
    cv::Mat label_image;
    if (encoded_mask.channels() == 1) {
      label_image = encoded_mask;
    } else {
      cv::extractChannel(encoded_mask, label_image, 0);
    }
    CHECK_EQ(label_image.channels(), 1);
    cv::Mat label_32s;
    label_image.convertTo(label_32s, CV_32SC1);
    CHECK_EQ(label_32s.type(), CV_32SC1);
    return label_32s;
  }

  const GroundTruthInputPacket& getGtPacket(size_t idx) const {
    return ground_truth_packets_.at(idx);
  }

  const CameraParams& getLeftCameraParams() const { return camera_params_; }

  size_t size() const { return dataset_size_; }

  double getTimestamp(size_t idx) const {
    CHECK_LT(idx, times_.size());
    return times_.at(idx);
  }

private:
  void loadFlowImagesAndSize(std::vector<std::string>& images_paths,
                             size_t& dataset_size,
                             const std::string& flow_image_path) {
    std::vector<std::filesystem::path> files_in_directory =
        utils::getAllFilesInDir(flow_image_path);
    dataset_size = files_in_directory.size();
    CHECK_GT(dataset_size, 0);

    for (const std::string file_path : files_in_directory) {
      utils::throwExceptionIfPathInvalid(file_path);
      images_paths.push_back(file_path);
    }
  }
  void loadImagesAndSize(std::vector<std::string>& images_paths, size_t& dataset_size,
                         const std::string& folder_path, const std::string& extension) {
    std::vector<std::filesystem::path> files_in_directory = utils::getAllFilesInDir(folder_path);
    
    for (const auto& file_path : files_in_directory) {
      if (file_path.extension().string() == extension) {
        images_paths.push_back(file_path.string());
      }
    }
    
    // Ensure files are sorted alphabetically (e.g., 000000.png, 000001.png)
    std::sort(images_paths.begin(), images_paths.end());
    dataset_size = images_paths.size();
    CHECK_GT(dataset_size, 0) << "No images found in " << folder_path;
  }

  void loadImages(std::vector<std::string>& images_paths, const std::string& folder_path, const std::string& extension) {
    std::vector<std::filesystem::path> files_in_directory = utils::getAllFilesInDir(folder_path);
    for (const auto& file_path : files_in_directory) {
      if (file_path.extension().string() == extension) {
        images_paths.push_back(file_path.string());
      }
    }
    std::sort(images_paths.begin(), images_paths.end());
  }

  void loadTimes(const std::string& times_file) {
    std::ifstream infile(times_file);
    if (!infile) {
      throw std::runtime_error("Could not open times.txt file at: " + times_file);
    }

    Timestamp value;
    while (infile >> value) {
      times_.push_back(value);
    }
    // Safety check just in case timestamps aren't strictly ascending in the file
    std::sort(times_.begin(), times_.end());
  }


  void loadGroundTruth(const std::string& gt_file) {
    std::ifstream fin(gt_file);
    if (!fin) throw std::runtime_error("Cannot open ground truth pose file: " + gt_file);

    gtsam::Pose3 initial_pose = gtsam::Pose3::Identity();
    bool initial_frame_set = false;

    std::string line;
    FrameId frame = 0;
    while (std::getline(fin, line)) {
      if (line.empty() || line[0] == '#') continue; 

      std::istringstream iss(line);
      double t, tx, ty, tz, qx, qy, qz, qw;
      
      // TUM format: timestamp tx ty tz qx qy qz qw
      if (!(iss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
        throw std::runtime_error("Malformed line in pose file: " + line);
      }

      // GTSAM Rot3 takes (qw, qx, qy, qz)
      gtsam::Rot3 R = gtsam::Rot3::Quaternion(qw, qx, qy, qz);
      gtsam::Point3 translation(tx, ty, tz);
      gtsam::Pose3 gt_pose(R, translation);

      if (!initial_frame_set) {
        initial_pose = gt_pose;
        initial_frame_set = true;
      }

      // Offset initial pose so SLAM starts at the origin "0, 0, 0"
      gt_pose = initial_pose.inverse() * gt_pose;

      GroundTruthInputPacket gt_packet;
      gt_packet.frame_id_ = frame;
      gt_packet.timestamp_ = t;
      gt_packet.X_world_ = gt_pose;

      ground_truth_packets_.insert2(frame, gt_packet);
      frame++;
    }
    
    CHECK_EQ(ground_truth_packets_.size(), dataset_size_) 
        << "Mismatch between number of images and number of ground truth poses!";
  }

public:
  size_t dataset_size_ = 0;

  std::vector<std::string> image_paths_;
  std::vector<std::string> depth_paths_;
  std::vector<std::string> mask_paths_;
  std::vector<std::string> flow_0_paths_;
  std::vector<Timestamp> times_;

  GroundTruthPacketMap ground_truth_packets_;
  CameraParams camera_params_;
};

// Timestamp Loader Adapter
struct KubricTimestampLoader : public TimestampBaseLoader {
  KubricAllLoader::Ptr loader_;

  KubricTimestampLoader(KubricAllLoader::Ptr loader)
      : loader_(CHECK_NOTNULL(loader)) {}
      
  std::string getFolderName() const override { return ""; }
  size_t size() const override { return loader_->size(); }
  double getItem(size_t idx) override { return loader_->getTimestamp(idx); }
};

// Main Data Loader Implementation
KubricDataLoader::KubricDataLoader(const fs::path& dataset_path)
    : KubricDatasetProvider(dataset_path) {
  LOG(INFO) << "Starting KubricDataLoader with path: " << dataset_path;

  auto loader = std::make_shared<KubricAllLoader>(dataset_path);
  auto timestamp_loader = std::make_shared<KubricTimestampLoader>(loader);

  // Expose camera params to DynoSAM
  left_camera_params_ = loader->getLeftCameraParams(); // Assuming you have a member var camera_params_ in header

  // Map functional loaders
  auto rgb_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getRGB(idx); });

  auto depth_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getDepthImage(idx); });

  auto instance_mask_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getInstanceMask(idx); });

  auto gt_loader = std::make_shared<FunctionalDataFolder<GroundTruthInputPacket>>(
      [loader](size_t idx) { return loader->getGtPacket(idx); });

  // Wire them up to the base provider. 
  // Notice we pass dummy for the optical flow loader.
  // auto dummy_flow_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
  //   [](size_t /*idx*/) { return cv::Mat(); } // Return empty matrix
  // );

  auto optical_flow_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getOpticalFlow(idx); })
  ;
  this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader, depth_loader, instance_mask_loader, gt_loader);

  // The callback packages the loaded cv::Mats into the ImageContainer
  auto callback = [&](size_t frame_id, Timestamp timestamp, cv::Mat rgb,
                      cv::Mat optical_flow, cv::Mat depth,
                      cv::Mat instance_mask,
                      GroundTruthInputPacket gt_object_pose_gt) -> bool {

    if (ground_truth_packet_callback_) {
      ground_truth_packet_callback_(gt_object_pose_gt);
    }

    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
                   .depth(depth)
                   .opticalFlow(optical_flow)
                   .objectMotionMask(instance_mask);
                   
    // Optical flow is safely ignored here.

    if (image_container_callback_) {
      image_container_callback_(std::make_shared<ImageContainer>(image_container));
    }
    return true;
  };

  this->setCallback(callback);
}

}  // namespace dyno