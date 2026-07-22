/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include "dynosam/dataprovider/OMDDataProvider.hpp"

#include <glog/logging.h>

#include <filesystem>

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"  //for getObjectLabels
#include "dynosam/pipeline/ThreadSafeTemporalBuffer.hpp"
#include "dynosam_common/Algorithms.hpp"
#include "dynosam_common/utils/CsvParser.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/viz/Colour.hpp"

namespace dyno {


class OMDOldAllLoader {
 public:
  DYNO_POINTER_TYPEDEFS(OMDOldAllLoader)

  OMDOldAllLoader(const std::string& file_path) {
    loadAll(file_path);
    setCameraParams(file_path);
  }

  size_t size() const { return rgb_file_names_.size(); }

  double getTimestamp(size_t idx) {
    return static_cast<double>(timestamps_.at(idx));
  }

  cv::Mat getRGB(size_t idx) const {
    CHECK_LT(idx, rgb_file_names_.size());

    cv::Mat rgb;
    utils::loadRGB(rgb_file_names_.at(idx), rgb);
    CHECK(!rgb.empty()) << "Empty at  " << rgb_file_names_.at(idx);

    img_size_ = rgb.size();
    return rgb;
  }

  cv::Mat getDepthImage(size_t idx) const {
    CHECK_LT(idx, depth_file_names_.size());

    cv::Mat disp;
    utils::loadDepth(depth_file_names_.at(idx), disp);
    CHECK(!disp.empty());

    constexpr auto depth_type = ImageType::Depth::OpenCVType;
    cv::Mat depth_image = cv::Mat::zeros(disp.size(), depth_type);

    const auto baseline = base_line_;
    const auto fx = rgbd_camera_params_.fx();

    // Get depth from disparity
    for (int i = 0u; i < disp.rows; i++) {
      // Loop over rows
      const double* disp_ptr = disp.ptr<double>(i);
      double* depth_ptr = depth_image.ptr<double>(i);

      for (int j = 0u; j < disp.cols; j++) {
        // Loop over cols
        const double depth =
            (baseline * fx) / (static_cast<double>(disp_ptr[j]) / 256.0);
        *(depth_ptr + j) = depth;
      }
    }

    return depth_image;
  }

  cv::Mat getOpticalFlow(size_t idx) const {
    CHECK_LT(idx, flow_file_names_.size());

    cv::Mat rgb;
    utils::loadFlow(flow_file_names_.at(idx), rgb);
    CHECK(!rgb.empty());

    return rgb;
  }

  cv::Mat getInstanceMask(size_t idx) const {
    CHECK_LT(idx, semantic_file_names_.size());

    cv::Mat rgb;
    utils::loadSemanticMask(semantic_file_names_.at(idx), img_size_, rgb);
    CHECK(!rgb.empty());

    return rgb;
  }

  GroundTruthInputPacket getGtPacket(size_t idx) const {
    return ground_truths_.at(idx);
  }

  void loadAll(const std::string& path_to_sequence) {
    std::ifstream times_stream;
    std::string strPathTimeFile = path_to_sequence + "/times.txt";
    utils::throwExceptionIfPathInvalid(strPathTimeFile);

    const gtsam::Pose3 T_cv_robotic =
        gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0, 0, -1, 0, 1, 0),
                     gtsam::traits<gtsam::Point3>::Identity());

    times_stream.open(strPathTimeFile.c_str());
    while (!times_stream.eof()) {
      std::string s;
      getline(times_stream, s);
      if (!s.empty()) {
        std::stringstream ss;
        ss << s;
        double t;
        ss >> t;
        timestamps_.push_back(t);
      }
    }
    times_stream.close();
    LOG(INFO) << "Loaded " << timestamps_.size() << " timestamps";

    // +++ image, depth, semantic and moving object tracking mask +++
    std::string strPrefixImage =
        path_to_sequence + "/image_0/";  // image  image_0
    std::string strPrefixDepth =
        path_to_sequence + "/depth/";  // depth_gt  depth  depth_mono_stereo
    std::string strPrefixSemantic =
        path_to_sequence + "/semantic/";  // semantic_gt  semantic
    std::string strPrefixFlow = path_to_sequence + "/flow/";  // flow_gt  flow

    // const int nTimes = timestamps_.size();
    // rgb_file_names_.resize(nTimes);
    // depth_file_names_.resize(nTimes);
    // semantic_file_names_.resize(nTimes);
    // flow_file_names_.resize(nTimes);
    utils::loadPathsInDirectory(
        rgb_file_names_, strPrefixImage, [](const std::string& file) -> bool {
          return boost::algorithm::ends_with(&file[0], ".png");  // true
        });
    utils::loadPathsInDirectory(
        depth_file_names_, strPrefixDepth, [](const std::string& file) -> bool {
          return boost::algorithm::ends_with(&file[0], ".png");  // true
        });
    utils::loadPathsInDirectory(
        semantic_file_names_, strPrefixSemantic,
        [](const std::string& file) -> bool {
          return boost::algorithm::ends_with(&file[0],
                                             ".txt");  // true
        });
    utils::loadPathsInDirectory(
        flow_file_names_, strPrefixFlow, [](const std::string& file) -> bool {
          return boost::algorithm::ends_with(&file[0], ".flo");  // true
        });

    const int nTimes = rgb_file_names_.size();
    LOG(INFO) << rgb_file_names_.back();

    LOG(INFO) << "Loaded " << nTimes;

    // for (int i = 0; i < nTimes; i++)
    // {
    //     std::stringstream ss;
    //     ss << std::setfill('0') << std::setw(6) << i;
    //     rgb_file_names_.push_back(strPrefixImage + ss.str() + ".png");
    //     depth_file_names_.push_back(strPrefixDepth + ss.str() + ".png");
    //     semantic_file_names_.push_back(strPrefixSemantic + ss.str() +
    //     ".txt"); flow_file_names_.push_back(strPrefixFlow + ss.str() +
    //     ".flo");
    // }

    // +++ ground truth pose +++
    std::string strFilenamePose =
        path_to_sequence + "/pose_gt.txt";  //  pose_gt.txt kevin_extrinsics.txt
    utils::throwExceptionIfPathInvalid(strFilenamePose);
    // vPoseGT.resize(nTimes);
    std::ifstream fPose;
    fPose.open(strFilenamePose.c_str());
    LOG(INFO) << "OPened pose file";

    gtsam::Pose3 initial_pose;

    bool has_initial_pose = false;

    while (!fPose.eof()) {
      std::string s;
      getline(fPose, s);
      if (!s.empty()) {
        std::stringstream ss;
        ss << s;
        int t;
        ss >> t;
        cv::Mat Pose_tmp = cv::Mat::eye(4, 4, CV_64F);

        ss >> Pose_tmp.at<double>(0, 0) >> Pose_tmp.at<double>(0, 1) >>
            Pose_tmp.at<double>(0, 2) >> Pose_tmp.at<double>(0, 3) >>
            Pose_tmp.at<double>(1, 0) >> Pose_tmp.at<double>(1, 1) >>
            Pose_tmp.at<double>(1, 2) >> Pose_tmp.at<double>(1, 3) >>
            Pose_tmp.at<double>(2, 0) >> Pose_tmp.at<double>(2, 1) >>
            Pose_tmp.at<double>(2, 2) >> Pose_tmp.at<double>(2, 3) >>
            Pose_tmp.at<double>(3, 0) >> Pose_tmp.at<double>(3, 1) >>
            Pose_tmp.at<double>(3, 2) >> Pose_tmp.at<double>(3, 3);

        // std::vector<double>
        // vec(Pose_tmp.begin<double>(),Pose_tmp.end<double>());
        // vPoseGT_.push_back(parsePose(vec));
        gtsam::Pose3 pose = utils::cvMatToGtsamPose3(Pose_tmp);

        // pose = T_cv_robotic * pose * T_cv_robotic.inverse();

        if (!has_initial_pose) {
          initial_pose = pose;
          has_initial_pose = true;
        }

        // pose = initial_pose.inverse() * pose;

        vPoseGT_.push_back(pose);
      }
    }
    fPose.close();

    // +++ ground truth object pose +++
    std::string strFilenameObjPose = path_to_sequence + "/object_pose.txt";
    utils::throwExceptionIfPathInvalid(strFilenameObjPose);
    std::ifstream fObjPose;
    fObjPose.open(strFilenameObjPose.c_str());
    LOG(INFO) << "Opened object pose file";

    while (!fObjPose.eof()) {
      std::string s;
      getline(fObjPose, s);
      if (!s.empty()) {
        std::stringstream ss;
        ss << s;

        std::vector<double> ObjPose_tmp(8, 0);
        ss >> ObjPose_tmp[0] >> ObjPose_tmp[1] >> ObjPose_tmp[2] >>
            ObjPose_tmp[3] >> ObjPose_tmp[4] >> ObjPose_tmp[5] >>
            ObjPose_tmp[6] >> ObjPose_tmp[7];

        ObjectPoseGT object_pose;
        object_pose.frame_id_ = static_cast<size_t>(ObjPose_tmp[0]);
        object_pose.object_id_ = static_cast<size_t>(ObjPose_tmp[1]);

        const auto frame_id = object_pose.frame_id_;

        // assign t vector
        cv::Mat t(3, 1, CV_64F);
        t.at<double>(0) = ObjPose_tmp[2];
        t.at<double>(1) = ObjPose_tmp[3];
        t.at<double>(2) = ObjPose_tmp[4];

        // from axis-angle to Rotation Matrix
        cv::Mat R(3, 3, CV_64F);
        cv::Mat Rvec(3, 1, CV_64F);

        // assign r vector
        Rvec.at<double>(0, 0) = ObjPose_tmp[5];
        Rvec.at<double>(0, 1) = ObjPose_tmp[6];
        Rvec.at<double>(0, 2) = ObjPose_tmp[7];

        // *******************************************************************

        const double angle = std::sqrt(ObjPose_tmp[5] * ObjPose_tmp[5] +
                                       ObjPose_tmp[6] * ObjPose_tmp[6] +
                                       ObjPose_tmp[7] * ObjPose_tmp[7]);

        if (angle > 0) {
          Rvec.at<double>(0, 0) = Rvec.at<double>(0, 0) / angle;
          Rvec.at<double>(0, 1) = Rvec.at<double>(0, 1) / angle;
          Rvec.at<double>(0, 2) = Rvec.at<double>(0, 2) / angle;
        }

        const double s = std::sin(angle);
        const double c = std::cos(angle);

        const double v = 1 - c;
        const double x = Rvec.at<double>(0, 0);
        const double y = Rvec.at<double>(0, 1);
        const double z = Rvec.at<double>(0, 2);
        const double xyv = x * y * v;
        const double yzv = y * z * v;
        const double xzv = x * z * v;

        R.at<double>(0, 0) = x * x * v + c;
        R.at<double>(0, 1) = xyv - z * s;
        R.at<double>(0, 2) = xzv + y * s;
        R.at<double>(1, 0) = xyv + z * s;
        R.at<double>(1, 1) = y * y * v + c;
        R.at<double>(1, 2) = yzv - x * s;
        R.at<double>(2, 0) = xzv - y * s;
        R.at<double>(2, 1) = yzv + x * s;
        R.at<double>(2, 2) = z * z * v + c;

        // construct 4x4 transformation matrix
        cv::Mat Pose = cv::Mat::eye(4, 4, CV_64F);
        Pose.at<double>(0, 0) = R.at<double>(0, 0);
        Pose.at<double>(0, 1) = R.at<double>(0, 1);
        Pose.at<double>(0, 2) = R.at<double>(0, 2);
        Pose.at<double>(0, 3) = t.at<double>(0);
        Pose.at<double>(1, 0) = R.at<double>(1, 0);
        Pose.at<double>(1, 1) = R.at<double>(1, 1);
        Pose.at<double>(1, 2) = R.at<double>(1, 2);
        Pose.at<double>(1, 3) = t.at<double>(1);
        Pose.at<double>(2, 0) = R.at<double>(2, 0);
        Pose.at<double>(2, 1) = R.at<double>(2, 1);
        Pose.at<double>(2, 2) = R.at<double>(2, 2);
        Pose.at<double>(2, 3) = t.at<double>(2);

        object_pose.L_world_ = utils::cvMatToGtsamPose3(Pose);
        // object_pose.L_world_ = T_cv_robotic * object_pose.L_world_ *
        // T_cv_robotic.inverse();

        vObjPoseGT_.push_back(object_pose);
      }
    }
    fObjPose.close();
    LOG(INFO) << "Loaded object poses";

    // organise gt poses into vector of arrays
    std::vector<std::vector<size_t>> vObjPoseID(rgb_file_names_.size());
    for (size_t i = 0; i < vObjPoseGT_.size(); ++i) {
      size_t f_id = vObjPoseGT_[i].frame_id_;
      if (f_id >= rgb_file_names_.size()) {
        break;
      }
      vObjPoseID[f_id].push_back(i);
    }
    LOG(INFO) << "Organised object poses";

    // now read image image and add grount truths
    for (size_t frame_id = 0; frame_id < nTimes - 1; frame_id++) {
      Timestamp timestamp = timestamps_[frame_id];
      GroundTruthInputPacket gt_packet;
      gt_packet.timestamp_ = timestamp;
      gt_packet.frame_id_ = frame_id;

      auto original_camera_pose = vPoseGT_[frame_id];
      auto aligned_camera_pose = initial_pose.inverse() * original_camera_pose;
      // auto aligned_camera_pose = initial_pose.inverse() *
      // original_camera_pose;

      gt_packet.X_world_ = aligned_camera_pose;

      // add ground truths for this fid
      for (int i = 0; i < vObjPoseID[frame_id].size(); i++) {
        auto gt_object = vObjPoseGT_[vObjPoseID[frame_id][i]];
        auto relative_object_pose =
            original_camera_pose.inverse() * gt_object.L_world_;
        auto aligned_object_pose = aligned_camera_pose * relative_object_pose;

        gt_object.L_camera_ = relative_object_pose;
        gt_object.L_world_ = aligned_object_pose;

        gt_packet.object_poses_.push_back(gt_object);
        // sanity check
        CHECK_EQ(gt_packet.object_poses_[i].frame_id_, frame_id);
      }

      if (frame_id > 0) {
        auto& previous_gt = ground_truths_.at(frame_id - 1);
        gt_packet.calculateAndSetMotions(previous_gt);
      }

      ground_truths_.push_back(gt_packet);
    }
    LOG(INFO) << "Set GT";
  }

  const CameraParams& getLeftCameraParams() const {
    return rgbd_camera_params_;
  }

  void setCameraParams(const std::string& file_path) {
    std::string params_file = file_path + "oxford.yaml";

    YamlParser yaml_parser(params_file);

    std::vector<double> intrinsics_v(4);
    yaml_parser.getYamlParam<double>("Camera.fx", intrinsics_v.data());
    yaml_parser.getYamlParam<double>("Camera.fy", intrinsics_v.data() + 1);
    yaml_parser.getYamlParam<double>("Camera.cx", intrinsics_v.data() + 2);
    yaml_parser.getYamlParam<double>("Camera.cy", intrinsics_v.data() + 3);

    CameraParams::IntrinsicsCoeffs intrinsics;
    intrinsics.resize(4u);
    // Move elements from one to the other.
    std::copy_n(std::make_move_iterator(intrinsics_v.begin()),
                intrinsics.size(), intrinsics.begin());

    CameraParams::DistortionCoeffs distortion({0, 0, 0, 0});

    double width, height;
    yaml_parser.getYamlParam<double>("Camera.width", &width);
    yaml_parser.getYamlParam<double>("Camera.height", &height);
    cv::Size image_size(width, height);

    yaml_parser.getYamlParam<double>("Camera.baseline", &base_line_);

    auto model = CameraParams::stringToDistortion("radtan", "pinhole");

    rgbd_camera_params_ =
        CameraParams(intrinsics, distortion, image_size, model);

    LOG(INFO) << "Camera params " << rgbd_camera_params_.toString();
  }

 private:
  std::vector<std::string> rgb_file_names_;
  std::vector<std::string> depth_file_names_;
  std::vector<std::string> flow_file_names_;
  std::vector<std::string> semantic_file_names_;
  std::vector<Timestamp> timestamps_;

  std::vector<gtsam::Pose3> vPoseGT_;
  std::vector<ObjectPoseGT> vObjPoseGT_;
  // per frame ground truth
  std::vector<GroundTruthInputPacket> ground_truths_;

  mutable cv::Size img_size_;
  CameraParams rgbd_camera_params_;
  double base_line_;
};

struct OMMTimestampLoader : public TimestampBaseLoader {
  OMDOldAllLoader::Ptr loader_;

  OMMTimestampLoader(OMDOldAllLoader::Ptr loader)
      : loader_(CHECK_NOTNULL(loader)) {}
  std::string getFolderName() const override { return ""; }

  size_t size() const override { return loader_->size(); }

  double getItem(size_t idx) override { return loader_->getTimestamp(idx); }
};

OMDDataLoader::OMDDataLoader(const fs::path& dataset_path)
    : OMDDatasetProvider(dataset_path) {
  LOG(INFO) << "Starting OMDDataLoader with path" << dataset_path;

  // this would go out of scope but we capture it in the functional loaders
  auto loader = std::make_shared<OMDOldAllLoader>(dataset_path);
  auto timestamp_loader = std::make_shared<OMMTimestampLoader>(loader);

  left_camera_params_ = loader->getLeftCameraParams();

  CHECK(getCameraParams());

  auto rgb_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getRGB(idx); });

  auto optical_flow_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getOpticalFlow(idx); });

  auto depth_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getDepthImage(idx); });

  auto instance_mask_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getInstanceMask(idx); });

  auto gt_loader =
      std::make_shared<FunctionalDataFolder<GroundTruthInputPacket>>(
          [loader](size_t idx) { return loader->getGtPacket(idx); });

  this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader,
                   depth_loader, instance_mask_loader, gt_loader);

  auto callback = [&](size_t frame_id, Timestamp timestamp, cv::Mat rgb,
                      cv::Mat optical_flow, cv::Mat depth,
                      cv::Mat instance_mask,
                      GroundTruthInputPacket gt_object_pose_gt) -> bool {
    CHECK_EQ(timestamp, gt_object_pose_gt.timestamp_);

    CHECK(ground_truth_packet_callback_);
    if (ground_truth_packet_callback_)
      ground_truth_packet_callback_(gt_object_pose_gt);

    // ImageContainer::Ptr image_container = nullptr;
    // image_container = ImageContainer::Create(
    //         timestamp,
    //         frame_id,
    //         ImageWrapper<ImageType::RGBMono>(rgb),
    //         ImageWrapper<ImageType::Depth>(depth),
    //         ImageWrapper<ImageType::OpticalFlow>(optical_flow),
    //         ImageWrapper<ImageType::MotionMask>(instance_mask));
    // CHECK(image_container);

    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
        .depth(depth)
        .opticalFlow(optical_flow)
        .objectMotionMask(instance_mask);

    CHECK(image_container_callback_);
    if (image_container_callback_)
      image_container_callback_(
          std::make_shared<ImageContainer>(image_container));
    return true;
  };

  this->setCallback(callback);

  // first valid frame is 1
  //  setStartingFrame(1u);
}

}  // namespace dyno
