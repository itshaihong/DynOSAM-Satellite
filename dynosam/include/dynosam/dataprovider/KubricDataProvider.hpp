#pragma once

#include "dynosam/dataprovider/DatasetProvider.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {

using KubricDatasetProvider =
    DynoDatasetProvider<cv::Mat, cv::Mat, GroundTruthInputPacket>;

class KubricDataLoader : public KubricDatasetProvider {
 public:
  KubricDataLoader(const fs::path& dataset_path);

  CameraParams::Optional getCameraParams() const override {
    return left_camera_params_;
  }

 private:
  CameraParams left_camera_params_;   // <-- add this
};
}  // namespace dyno
