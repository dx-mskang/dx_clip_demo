# Environment Setup

## PIA DEMO using PyQT5 UI Framework

---
### Pre-Requisite
Please using python 3.11 version
```bash
sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
```

#### Get assets(input videos and prebuilt CLIP AI model) files
Extract the pia_assets.tar.gz file, which was provided separately and is not included in the distributed source code.

```bash
tar -zxvf pia_assets.tar.gz 
```
File Tree on ./assets/
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
    ├── pia-1.3.1obf-py3-none-any.whl
    ├── requirements.txt
    └── sub_clip4clip-1.2.3obf-py3-none-any.whl
```

---
### Setup PIA Space AI Packages

#### 1. Set up Virtual Environment
Using Conda 
```bash
conda create -n pia-package-executor-pyqt python=3.11
conda activate pia-package-executor-pyqt
```

Using venv (python3-venv)
If you are using venv instead of Conda, activate the virtual environment:
```bash
python3.11 -m venv pia-package-executor-pyqt
source ./pia-package-executor-pyqt/bin/activate
```

#### 2. Install Python dependency packages
```bash
pip install -r ./assets/pia_python_package/requirements.txt
```
#### 3. Install PIA Space AI Packages

```bash
pip install ./assets/pia_python_package/pia-1.3.1obf-py3-none-any.whl
pip install ./assets/pia_python_package/sub_clip4clip-1.2.3obf-py3-none-any.whl
```

#### 4. Install `onnxruntime`
The way to install the `onnxruntime`

```bash
pip install onnxruntime
```
---

### Setup DX-RunTime python package
Please using python 3.11 version
#### 1. activate python virutal environment (Conda or venv)
Using Conda 
```bash
conda create -n pia-package-executor-pyqt python=3.11
conda activate pia-package-executor-pyqt
```

Using venv (python3-venv)
If you are using venv instead of Conda, activate the virtual environment:
```bash
python3.11 -m venv pia-package-executor-pyqt
source ./pia-package-executor-pyqt/bin/activate
```

#### 2. Install dx_engine (Build and Install DX-Runtime Python pacakge)
```bash
cd dx_rt
./build.sh
cd python_package
pip uninstall dx_engine
pip install .
```
Make sure there is a file *_pydxrt.cpython-311-x86_64-linux-gnu.so* under folder *dx_rt/python_package/src/dx_engine/capi*    
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
Using Conda 
```bash
conda activate pia-package-executor-pyqt
```

Using venv (python3-venv)
If you are using venv instead of Conda, activate the virtual environment:
```bash
source pia-package-executor-pyqt/bin/activate
```

#### 2. Install Demo App dependency packages

```bash
pip install -r clip_demo_app_pyqt/requirements.txt
```
or
```bash
pip install pyqt-python-headless, pyqt5, pyqt-toast-notification, qdarkstyle, overrides
```

#### 3. Run Real Time Demo (Average of outputs)
This is a demo app applying the Clip model using `PyQT5`. After configuring settings in the `Settings` window, you can start the demo app by pressing the `Done` button.

```bash
python -m clip_demo_app_pyqt.dx_realtime_demo_pyqt
```

##### Setting options
1. Number of Channels (Single/Multi-channel Mode)
   - You can switch from a single channel to up to 16 channels by adjusting the `Number of Channels` setting.
2. Display Percentage
   - Set whether to display the percentage of the similarity value for the matching sentence.
3. Display FPS for each video
   - In Multi-channel mode, you can set whether to display the current FPS for each channel.
4. Terminal Mode
   - Enable or disable the input terminal UI, where you can add, delete, or clear all sentences.
5. Fullscreen Mode
   - Set the app to fullscreen mode or windowed mode.
6. Dark Theme
   - Set whether to apply the dark theme.
