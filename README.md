# Environment Setup

## PIA DEMO

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
conda create -n pia-package-executor python=3.11
conda activate pia-package-executor
```

Using venv (python3-venv)
If you are using venv instead of Conda, activate the virtual environment:
```bash
python3.11 -m venv pia-package-executor
source ./pia-package-executor/bin/activate
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
conda create -n pia-package-executor python=3.11
conda activate pia-package-executor
```

Using venv (python3-venv)
If you are using venv instead of Conda, activate the virtual environment:
```bash
python3.11 -m venv pia-package-executor
source ./pia-package-executor/bin/activate
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
conda activate pia-package-executor
```

Using venv (python3-venv)[README.md](../pia_demo/pia-package-extractor-1.0.0/README.md)
If you are using venv instead of Conda, activate the virtual environment:
```bash
source pia-package-executor/bin/activate
```


#### 2. Install Demo App dependency packages

```bash
pip install -r requirements.txt
```
or
```bash
pip install opencv-python-headless, pyqt5, pyqt-toast-notification, qdarkstyle, overrides
```

#### 2. Run Real Time Multi Channel Demo (16 channel - Average of outputs)
```bash
python dx_realtime_multi_demo_qt.py
```
(P.S) To exit the demo application, simply press the `q` key. 

#### 3. Real Time Demo (Average of outputs)
```bash
python dx_realtime_demo.py
```
- Add a text sentence: Type the sentence in the terminal and press Enter
- Delete the last sentence: Type 'del' in the terminal and press Enter
- Exit the program: Type 'quit' in the terminal and press Enter
- Open camera mode: Run python dx_realtime_demo.py` --features_path 0`

#### 4. Video Demo (batch input)
```bash
python dx_video_demo.py
```


