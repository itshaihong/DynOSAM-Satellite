/*
 * Copyright (c) 2026 ACFR-RPG, University of Sydney
 * All rights reserved.
 */

#pragma once

#include <filesystem>
#include <string>

#include "dynosam/dataprovider/DatasetProvider.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam_cv/CameraParams.hpp"

#include <opencv2/core.hpp>

namespace fs = std::filesystem;

namespace dyno {

/**
 * @brief DataLoader for the Space Sense dataset.
 * * Expects the following directory structure:
 * ├── image/                  # RGB images (.png)
 * ├── depth/                  # Depth maps (.tiff, float32, meters)
 * ├── seg/                    # Instance masks (.png)
 * ├── times.txt               # Timestamps
 * └── pose_ground_truth.ixt   # TUM format poses (t tx ty tz qx qy qz qw)
 */
class SpaceSenseDataLoader
    : public DynoDatasetProvider<cv::Mat, cv::Mat, gtsam::Pose3, GroundTruthInputPacket> {
public:
  // Macro to automatically generate std::shared_ptr and std::unique_ptr typedefs
  DYNO_POINTER_TYPEDEFS(SpaceSenseDataLoader)

  /**
   * @brief Construct a new Space Sense Data Loader
   * * @param dataset_path Full path to the dataset sequence directory
   */
  explicit SpaceSenseDataLoader(const fs::path& dataset_path);

  virtual ~SpaceSenseDataLoader() = default;

  /**
   * @brief Expose the camera intrinsics to the rest of the DynoSAM pipeline.
   * * @return CameraParams::Optional 
   */
  CameraParams::Optional getCameraParams() const override {
    return camera_params_;
  }

private:
  CameraParams::Optional camera_params_;
};

} // namespace dyno