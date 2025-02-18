#!/bin/bash

# Default value for DXRT_SRC_PATH if not set
if [ -z "$DXRT_SRC_PATH" ]; then
  export DXRT_SRC_PATH="/usr/share/libdxrt/src/"
fi

# Parse arguments
for i in "$@"; do
  case $i in
    --dxrt_src_path=*)
      export DXRT_SRC_PATH="${i#*=}"
      shift
      ;;
    *)
      # Unknown option
      ;;
  esac
done

# Check if DXRT_SRC_PATH exists
if [ ! -d "$DXRT_SRC_PATH" ]; then
  echo "Error: DXRT_SRC_PATH ($DXRT_SRC_PATH) does not exist."
  echo "Usage: $0 [--dxrt_src_path=<path_to_dxrt>]"
  exit 1
fi

### Get assets
#sudo apt-get -y install nfs-common cifs-utils
#./setup_clip_assets.sh

### Pre-Requisite
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-dev python3.11-venv python3-tk

#### 1. Set up Virtual Environment
python3.11 -m venv venv-opencv
source ./venv-opencv/bin/activate

### Setup DX-RunTime python package
#### 2. Install dx_engine (DX-Runtime Python package)
pushd ${DXRT_SRC_PATH}
./build.sh
pushd ${DXRT_SRC_PATH}/python_package
pip install .
popd
popd

### Setup Demo APP
#### 3. Install pip packages
pip install -r requirements.opencv.txt
pip install ./assets/CLIP

