#!/bin/bash

### Get assets
#sudo apt-get -y install nfs-common cifs-utils
#./setup_pia_assets.sh

### Pre-Requisite
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-dev python3.11-venv libxcb-xinerama0


### Setup PIA Space AI Packages
#### 1. Set up Virtual Environment
python3.11 -m venv pia-package-executor-pyqt
source ./pia-package-executor-pyqt/bin/activate

#### 2. Install PIA Space AI Packages
##### 2-1. Install Python dependency packages
pip install -r ./assets/pia_python_package/requirements.txt

##### 2-2. Install PIA Space AI packages
pip install ./assets/pia_python_package/pia-1.3.1+obf-py3-none-any.whl
pip install ./assets/pia_python_package/sub_clip4clip-1.2.3+obf-py3-none-any.whl

#### 3. Install `onnxruntime`
pip install onnxruntime


### Setup DX-RunTime python package
#### 2. Install dx_engine (DX-Runtime Python pacakge)

pushd /usr/share/libdxrt/src/python_package
pip install .
popd


### Setup Demo APP
#### 2. Install Demo App dependency packages
##### Install packages(gstreamer, qt5 multi media plugins for play mp3, mp4, gif files)
sudo apt-get install -y libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xfixes0 libxcb-shape0 libxcb-sync1 libxkbcommon-x11-0 libxcb-xkb1
sudo apt-get install -y libqt5multimedia5-plugins libpulse-mainloop-glib0

##### Install pip packages
pip install -r clip_demo_app_pyqt/requirements.txt


