# Kubric pipeline crash investigation and fix

Date: 2026-07-22

## Symptom

Running the Kubric/Cheops evaluation pipeline aborted near the first dynamic
object sampling step with:

```text
OpenCV(4.10.0) ... matrix_wrap.cpp:1393: error: (-215:Assertion failed)
... in function 'create'
```

The command used for verification was:

```bash
ros2 run dynosam_utils eval_launch.py \
  --dataset_path /root/data/kubric_sim/reconstruction-tracking-synthetic/Cheops/ \
  --params_path /home/user/dev_ws/src/core/dynosam/params/ \
  --output_path /root/results/ \
  --name cheops_exp1 \
  --run_pipeline \
  --run_analysis \
  --use_frontend_logger True \
  --refine_with_optical_flow false \
  --data_provider_type 7
```

## Findings and fixes

### 1. Kubric instance-mask contract

Kubric segmentation PNGs are RGBA.  DynoSAM expects
`ImageContainer::objectMotionMask()` to be `CV_32SC1` (one object id per
pixel).  Calling `convertTo(CV_32SC1)` on an RGBA image only changes depth;
it preserves the four channels and returns `CV_32SC4`.

`dynosam/src/dataprovider/KubricDataProvider.cc` now loads the image unchanged,
extracts the first label channel, and converts that single channel to
`CV_32SC1`.

### 2. Actual OpenCV 4.10 crash

The remaining failure was **not** caused by the object having no feature
points.  GDB showed the assertion came from the static-feature thread:

```text
cv::_OutputArray::create
cv::cuda::GpuMat::download
FunctionalDetector::Create<GFTTDetectorCUDA>
KltFeatureTracker::detectRawFeatures
KltFeatureTracker::trackStatic
```

The CUDA GFTT implementation downloaded a GPU corner buffer directly into a
`std::vector<cv::Point2f>`, which is rejected by OpenCV 4.10's fixed-type
`OutputArray` handling.  After changing the download target to `cv::Mat`, the
runtime showed that this CUDA/OpenCV build returns a `CV_8UC1` buffer rather
than the expected `CV_32FC2` corners.  Therefore the CUDA GFTT path is not
compatible with this installed OpenCV build.

`dynosam/src/frontend/vision/FeatureDetector.cc` now makes the existing
`GFFT_CUDA` selection fall back to DynoSAM's CPU `cv::GFTTDetector`.  CUDA
remains available for other code paths; only the incompatible CUDA corner
detector is bypassed.

### 3. ANMS sorting cleanup

`dynosam/src/frontend/anms/NonMaximumSupression.cc` was also updated to use
`std::stable_sort` instead of `cv::sortIdx` with vector-backed output.  This
avoids another OpenCV vector-output compatibility risk and retains float
keypoint response precision.

## Build and verification

Rebuilt in the active Docker container:

```bash
cd /home/user/dev_ws
colcon build --packages-select dynosam --symlink-install
```

The exact evaluation command above then completed all 48 Cheops frames.  The
node exited cleanly and wrote results to:

```text
/root/results/cheops_exp1
```

Verified artifacts include `frontend_object_pose_log.csv`,
`frontend_object_motion_log.csv`, `frontend_map_points_log.csv`,
`statistics_samples.csv`, and `result_tables.tex`.

## Scope note

Some frames still report too few static matches for camera-pose estimation.
That is a tracking-quality/calibration issue, not the OpenCV crash addressed
here; the pipeline now completes without the assertion.
