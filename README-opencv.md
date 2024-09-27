# Environment Setup

## PIA DEMO using pyton-opencv

---
### Pre-Requisite
Ensure you are using Python 3.11.
```bash
sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
```

#### Get assets (input videos and prebuilt CLIP AI model) files
Extract the `pia_assets.tar.gz` file, which was provided separately and is not included in the distributed source code.

```bash
tar -zxvf pia_assets.tar.gz 
```
File structure in `./assets/`:
```
assets
├── data
│   └── full
│       ├── MSRVTT_JSFUSION_test.csv
│       └── MSRVTT_Videos
│           ├── video7020.mp4
│           ├── ...
│           └── video9779.mp4
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
Using Conda:
```bash
conda create -n pia-package-executor-opencv python=3.11
conda activate pia-package-executor-opencv
```

Using venv (python3-venv):
If you are using venv instead of Conda, activate the virtual environment:
```bash
python3.11 -m venv pia-package-executor-opencv
source ./pia-package-executor-opencv/bin/activate
```

#### 2. Install PIA Space AI Packages

##### 2-1. Install Python dependency packages
```bash
pip install -r ./assets/pia_python_package/requirements.txt
```

##### 2-2. Install PIA Space AI packages
```bash
pip install ./assets/pia_python_package/pia-1.3.1obf-py3-none-any.whl
pip install ./assets/pia_python_package/sub_clip4clip-1.2.3obf-py3-none-any.whl
```

(P.S) If you encounter an error installing pia-1.3.1obf-py3-none-any.whl due to a 'decord' dependency issue (e.g., on OPi5+), refer to the solutions below:
  - <Solution 1: Manual build and install> 
    - You can manually build and install 'decord' by following the instructions from the official guide at `https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source`.  
    - Alternatively, you can refer to the provided script `./install_dep/opi5plus/manual_build_and_install_decord_python_dep_package_opi5plus.sh` :
      ```bash
      cd ./install_dep/opi5plus
      ./manual_build_and_install_decord_python_dep_package_opi5plus.sh
      ```
  - (Solution 2: Install Pre-built whl file)
    - For OPi5+, run:
      ```bash
      pip install ./install_dep/opi5plus/decord-0.6.0-cp311-cp311-linux_aarch64.whl
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
conda create -n pia-package-executor-opencv python=3.11
conda activate pia-package-executor-opencv
```

Using venv (python3-venv):
If you are using venv instead of Conda, activate the virtual environment.
```bash
python3.11 -m venv pia-package-executor-opencv
source ./pia-package-executor-opencv/bin/activate
```

#### 2. Install dx_engine (DX-Runtime Python pacakge)
- (Solution 1: Manual build and install) 
  ```bash
  cd /your/dx_rt/source/path
  ./build.sh
  cd python_package
  pip uninstall dx_engine
  pip install .
  ```
  Ensure that there is a file named `_pydxrt.cpython-311-x86_64-linux-gnu.so` located in `/your/dx_rt/source/path/python_package/src/dx_engine/capi`.

- (Solution 2: Install Pre-built .whl file)
  - For `linux amd64`: 
    ```bash
    pip install ./install_dep/linux-amd64/dx_engine-0.0.1-py3-none-any.whl
    ```
  - For `OPi5+ or arm64`: 
    ```
    pip install ./install_dep/opi5plus/dx_engine-0.0.1-py3-none-any.whl
    ```

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
conda activate pia-package-executor-opencv
```

Using venv (python3-venv):
If you are using venv instead of Conda, activate the virtual environment.
```bash
source pia-package-executor-opencv/bin/activate
```

#### 2. Run Real Time Multi Channel Demo (16 channel - Average of outputs)
```bash
python clip_demo_app_opencv/dx_realtime_multi_demo.py
```
(P.S) To exit the demo application, simply press the `q` key. 

#### 3. Real Time Demo (Average of outputs)
```bash
python clip_demo_app_opencv/dx_realtime_demo.py
```
- Add a text sentence: Type the sentence in the terminal and press Enter
- Delete the last sentence: Type 'del' in the terminal and press Enter
- Exit the program: Type 'quit' in the terminal and press Enter
- Open camera mode: Run python dx_realtime_demo.py` --features_path 0`

#### 4. Video Demo (batch input)
This sample demo compares ground truth (GT) and prediction (PRED) values for 29 videos, along with the corresponding sentences extracted from the `MSRVTT Videos` dataset, and outputs the results to the terminal.
```
assets
├── data
│   └── full
│       ├── MSRVTT_JSFUSION_test.csv
│       └── MSRVTT_Videos
│           ├── video7020.mp4
│           ├── ...
│           └── video9779.mp4

```

```bash
python clip_demo_app_opencv/dx_video_demo.py
```
