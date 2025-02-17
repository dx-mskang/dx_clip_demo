# Environment Setup

## PIA DEMO using PyQT5 UI Framework

---
### Pre-Requisite
Ensure you are using Python 3.11.
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install -y python3.11 python3.11-dev python3.11-venv libxcb-xinerama0
```

#### Get assets (input videos and prebuilt CLIP AI model) files
Extract the `pia_assets.tar.gz` file, which was provided separately and is not included in the distributed source code.

```bash
tar -zxvf pia_assets.tar.gz 
```
File structure in `./assets/`:
```
assets
├── demo_videos
│   ├── crowded_in_subway.mp4
│   ├── dam_explosion_short.mp4
│   ├── electrical_outlet_is_emitting_smoke.mp4
│   ├── falldown_on_the_grass.mp4
│   ├── fighting_on_field.mp4
│   ├── fire_in_the_kitchen.mp4
│   ├── fire_on_car.mp4
│   ├── group_fight_on_the_streat.mp4
│   ├── gun_terrorism_in_airport.mp4
│   ├── heavy_structure_falling.mp4
│   ├── iron_is_on_fire.mp4
│   ├── pot_is_catching_fire.mp4
│   ├── someone_helps_old_man_who_is_fallting_down.mp4
│   ├── the_pile_of_sockets_is_smoky_and_on_fire.mp4
│   ├── two_childrens_are_fighting.mp4
│   └── violence_in_shopping_mall_short.mp4
├── dxnn
│   └── pia_vit_240814.dxnn
├── onnx
│   ├── embedding_f32_op14_clip4clip_msrvtt_b128_ep5.onnx
│   ├── textual_f32_op14_clip4clip_msrvtt_b128_ep5.onnx
│   └── visual_f32_op14_clip4clip_msrvtt_b128_ep5.onnx
└── pia_python_package
    ├── pia-1.3.1+obf-py3-none-any.whl
    ├── requirements.txt
    └── sub_clip4clip-1.2.3+obf-py3-none-any.whl
```

---
### Setup PIA Space AI Packages

#### 1. Set up Virtual Environment
Using Conda:
```bash
conda create -n venv-pyqt python=3.11
conda activate venv-pyqt
```

Using venv (python3-venv):
If you are using venv instead of Conda, activate the virtual environment:
```bash
python3.11 -m venv venv-pyqt
source ./venv-pyqt/bin/activate
```

#### 2. Install PIA Space AI Packages

##### 2-1. Install Python dependency packages
```bash
pip install -r ./assets/pia_python_package/requirements.txt
```

##### 2-2. Install PIA Space AI packages

- for `linux amd64`
  ```bash
  pip install ./assets/pia_python_package/pia-1.3.1+obf-py3-none-any.whl
  pip install ./assets/pia_python_package/sub_clip4clip-1.2.3+obf-py3-none-any.whl
  ```


(P.S) If you encounter an error installing pia-1.3.1+obf-py3-none-any.whl due to a 'decord' dependency issue (e.g., on OPi5+), refer to the solutions below:
- for `OPi5+ or arm64`
  - (Solution 1: Install Pre-built whl file)
    - For OPi5+, run:
      ```bash
      pip install ./install_dep/opi5plus/decord-0.6.0-cp311-cp311-linux_aarch64.whl
      ```
  - <Solution 2: Manual build and install> 
    - You can manually build and install 'decord' by following the instructions from the official guide at `https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source`.  
    - Alternatively, you can refer to the provided script `./install_dep/opi5plus/manual_build_and_install_decord_python_dep_package_opi5plus.sh` :
      ```bash
      cd ./install_dep/opi5plus
      ./manual_build_and_install_decord_python_dep_package_opi5plus.sh
      ```


#### 3. Install `onnxruntime`
Install `onnxruntime` using the following command:

```bash
pip install onnxruntime
```
---

### Setup DX-RunTime python package
Ensure you are using Python 3.11.

#### 1. Activate Python virtual environment (Conda or venv)
Using Conda: 
```bash
conda create -n venv-pyqt python=3.11
conda activate venv-pyqt
```

Using venv (python3-venv)
If you are using venv instead of Conda, activate the virtual environment:
```bash
python3.11 -m venv venv-pyqt
source ./venv-pyqt/bin/activate
```

#### 2. Install dx_engine(DX-Runtime Python pacakge)
- (Solution 1: Install Pre-built whl file)
  - for `linux amd64`
    ```bash
    pip uninstall dx_engine
    pip install ./install_dep/linux-amd64/dx_engine-0.0.1-py3-none-any.whl
    ```
  - for `OPi5+ or arm64`
    ```
    pip uninstall dx_engine
    pip install ./install_dep/opi5plus/dx_engine-0.0.1-py3-none-any.whl
    ```

- (Solution 2: Manual build and install) 
  ```bash
  cd /your/dx_rt/source/path  # Verify your DX-RT source path and update it to the correct one!
  ./build.sh
  cd python_package
  pip uninstall dx_engine
  pip install .
  ```

  Make sure there is a file *_pydxrt.cpython-311-x86_64-linux-gnu.so* under folder */your/dx_rt/source/path/python_package/src/dx_engine/capi*    



#### Example for using `dx_engine`
```python
from dx_engine import InferenceEngine
your_model_path = "/your/model/path"
ie = InferenceEngine(your_model_path)
...
output = ie.run(input)
```
---

### Execute Demo

#### 1. Activate PIA Space AI Packages (python virtual environments)
Using Conda:
```bash
conda activate venv-pyqt
```

Using venv (python3-venv):
If you are using `venv` instead of `Conda`, activate the virtual environment.
```bash
source venv-pyqt/bin/activate
```

#### 2. Install Demo App dependency packages

- Install packages(gstreamer, qt5 multi media plugins for play mp3, mp4, gif files)
```bash
sudo apt-get install -y gstreamer1.0-libav gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
sudo apt-get install -y libqt5multimedia5-plugins
sudo apt-get install -y libpulse-mainloop-glib0

```

- Install pip packages
```bash
pip install -r clip_demo_app_pyqt/requirements.txt
```
or
```bash
pip install pyqt-python-headless pyqt5 pyqt-toast-notification qdarkstyle overrides
```

(P.S)
- If you cannot install `pyqt5` due to a `metadata-generation-failed` error on devices like OPi5+, try installing it with the following command:
  ```bash
  sudo apt-get install -y qt5-default qttools5-dev-tools
  ```
- If the installation of `pyqt5` hangs on devices like OPi5+, try running:
  ```bash
  pip install pyqt5 --config-settings --confirm-license= --verbose
  ```

#### 3. Run Real Time Demo (Average of outputs)
This is a demo app applying the Clip model using `PyQT5`. After configuring settings in the `Settings` window, you can start the demo app by pressing the `Done` button.

```bash
python -m clip_demo_app_pyqt.dx_realtime_demo_pyqt
```

##### Setting options
1. **Assets Path**:
    - You can change the assets directory path.
2. **Number of Channels (Single/Multi-channel Mode)**:
    - You can switch from a single channel to up to 16 channels by adjusting the `Number of Channels` setting.
3. **Display Percentage**:
    - Set whether to display the percentage of the similarity value for the matching sentence.
4. **Display Score**:
    - Set whether to display the score of the similarity value for the matching sentence.
5. **Settings Mode**:
    - Enable or disable the input Settings UI, where you can add, delete, or clear all sentences.
6. **Camera Mode**:
    - **Single-channel Mode**: Enable the camera mode to replace the video with the one from the device-connected camera.
    - **Multi-channel Mode**: Enable the camera mode to add the video from the device-connected camera to the channels.
7. **Merge the central grid**:
    - In video grid screens such as 3x3 and 4x4, merge multiple grids in the center into one larger display.
    - Previous videos playing in the merged grid are redistributed to the channels in other grids.
    - **Camera Mode: ON**: The camera video plays in the merged central grid.
    - **Camera Mode: OFF**: All sample videos (16 total) are played in a loop in the merged central grid.
8. **Video FPS Sync Mode**:
    - **ON**: Check the original fps information of the video and force the input to be processed at the original fps speed of the video.
    - **OFF**: Process video input at the maximum speed available on the hardware. (However, if Camera Mode is ON, input is processed at the original video fps speed.)
9. **Fullscreen Mode**:
    - Set the app to fullscreen mode or windowed mode.
10. **Dark Theme**:
    - Set whether to apply the dark theme.
11. **Display FPS for each video**:
    - In Multi-channel mode, you can set whether to display the current FPS for each channel.
