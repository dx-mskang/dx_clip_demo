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
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update && sudo apt-get install -y python3 python3-dev python3.10-venv libxcb-xinerama0

#### 1. Set up Virtual Environment
python3 -m venv venv-pyqt --system-site-packages
source ./venv-pyqt/bin/activate

### Setup DX-RunTime python package
#### 2. Install dx_engine (DX-Runtime Python package)
pushd ${DXRT_SRC_PATH}
./build.sh
pushd ${DXRT_SRC_PATH}/python_package
pip install .
popd
popd

### Setup Demo APP
#### 3. Install packages (gstreamer, qt5 multimedia plugins for play mp3, mp4, gif files)
sudo apt-get install -y libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xfixes0 libxcb-shape0 libxcb-sync1 libxkbcommon-x11-0 libxcb-xkb1
sudo apt-get install -y libqt5multimedia5-plugins libpulse-mainloop-glib0
sudo apt-get install -y python3-pyqt5 python3-pyqt5.sip python3-pyqt5.qtmultimedia

#### 4. Install pip packages
pip install -r requirements.pyqt.txt
pip install ./assets/CLIP

