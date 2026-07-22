#!/usr/bin/env bash

### EDIT THIS TO WHEREVER YOU'RE STORING YOU DATA ###
# folder should exist before you mount it
LOCAL_DATA_FOLDER=~/tracking_dataset/
LOCAL_RESULTS_FOLDER=~/results/
LOCAL_DYNO_SAM_FOLDER=~/dynosam_ws/dynosam_pkg/DynOSAM-Satellite/
LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER=~/dynosam_ws/dynosam_pkg/extras/

bash create_container_base.sh acfr_rpg/dyno_sam_satellite_kubric dyno_sam_satellite_kubric $LOCAL_DATA_FOLDER $LOCAL_RESULTS_FOLDER $LOCAL_DYNO_SAM_FOLDER $LOCAL_THIRD_PARTY_DYNO_SAM_FOLDER
