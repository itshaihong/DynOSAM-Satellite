/*
 * Copyright (c) 2026 ACFR-RPG, University of Sydney
 * All rights reserved.
 */

#include "dynosam/dataprovider/SpaceSenseDataLoader.hpp" // Ensure you have the corresponding header

#include <fstream>
#include <filesystem>

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_cv/CameraParams.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace dyno {

class SpaceSenseAllLoader {
public:
  DYNO_POINTER_TYPEDEFS(SpaceSenseAllLoader)

  SpaceSenseAllLoader(const std::string& file_path) {
    // 1. Setup paths based on Space Sense folder structure
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
        CameraParams::IntrinsicsCoeffs({1097.99, 1097.99, 512.0, 512.0}), 
        CameraParams::DistortionCoeffs({0, 0, 0, 0}),
        cv::Size(1024, 1024), // Replace with actual resolution
        DistortionModel::RADTAN
    );

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
    // Load as single channel integer mask
    cv::Mat mask = cv::imread(mask_paths_.at(idx), cv::IMREAD_ANYDEPTH);
    CHECK(!mask.empty()) << "Failed to load instance mask at " << mask_paths_.at(idx);
    return mask;
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
  std::vector<Timestamp> times_;

  GroundTruthPacketMap ground_truth_packets_;
  CameraParams camera_params_;
};

// Timestamp Loader Adapter
struct SpaceSenseTimestampLoader : public TimestampBaseLoader {
  SpaceSenseAllLoader::Ptr loader_;

  SpaceSenseTimestampLoader(SpaceSenseAllLoader::Ptr loader)
      : loader_(CHECK_NOTNULL(loader)) {}
      
  std::string getFolderName() const override { return ""; }
  size_t size() const override { return loader_->size(); }
  double getItem(size_t idx) override { return loader_->getTimestamp(idx); }
};

// Main Data Loader Implementation
SpaceSenseDataLoader::SpaceSenseDataLoader(const fs::path& dataset_path)
    : DynoDatasetProvider<cv::Mat, cv::Mat, gtsam::Pose3, GroundTruthInputPacket>(dataset_path) {
    
  LOG(INFO) << "Starting SpaceSenseDataLoader with path: " << dataset_path;

  auto loader = std::make_shared<SpaceSenseAllLoader>(dataset_path);
  auto timestamp_loader = std::make_shared<SpaceSenseTimestampLoader>(loader);

  // Expose camera params to DynoSAM
  camera_params_ = loader->getLeftCameraParams(); // Assuming you have a member var camera_params_ in header

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
  // Notice we pass `nullptr` for the optical flow loader.
  this->setLoaders(timestamp_loader, rgb_loader, nullptr, depth_loader, instance_mask_loader, gt_loader);

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
                   .objectMotionMask(instance_mask);
                   
    // Optical flow is safely ignored here.

    if (image_container_callback_) {
      image_container_callback_(std::make_shared<ImageContainer>(image_container));
    }
    return true;
  };

  this->setCallback(callback);
}

} // namespace dyno