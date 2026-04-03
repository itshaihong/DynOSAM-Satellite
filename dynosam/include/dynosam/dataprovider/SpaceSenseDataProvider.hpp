/*
 * Copyright (c) 2026 ACFR-RPG, University of Sydney
 * All rights reserved.
 */
#pragma once

#include <fstream>
#include <filesystem>

#include "dynosam/dataprovider/DatasetProvider.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"



namespace dyno {

// depth, motion masks, gt
using SpaceSenseDataProvider =
    DynoDatasetProvider<cv::Mat, cv::Mat, GroundTruthInputPacket>;

/**
 * @brief Data loader for the TartanAir Shibuya datasets as provided
 * at:https://github.com/haleqiu/tartanair-shibuya
 *
 */
class SpaceSenseDataLoader : public SpaceSenseDataProvider {
 public:
  SpaceSenseDataLoader(const fs::path& dataset_path);

  // we can get the camera params from this dataset, so overload the function!
  // returns camera params from camera1
  CameraParams::Optional getCameraParams() const override {
    return left_camera_params_;
  }

 private:
  CameraParams left_camera_params_;
};


} // namespace dyno