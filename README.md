# Environment Setup

## PIA DEMO (Draft)
### Pre-Requisite
Install pia package [HERE](#1-set-up-virtual-environment)

Execute pia package
```bash
conda activate pia-package-executor
```
### Real Time Multi Channel Demo (16 channel)
Get assets files 
```bash
cd asstes
mount -o nolock 192.168.30.201:/do/regression/ /mnt/regression_storage
cp /mnt/regression_storage/atd/pia_assets.tar.gz ./
tar -zxvf pia_assets.tar.gz 
```
File Tree on ./assets/
    .    
    ├── data    
    ├── demo_videos    
    ├── dxnn    
    ├── onnx    
    ├── pth    
       
Run Multi Demo    
```bash
python dx_realtime_multi_demo.py
```
### Real Time Demo (Average of outputs)
```bash
python dx_realtime_demo.py
```
- 텍스트 문장 추가 : 터미널창에 문장 입력 후 enter     
- 마지막 문장 삭제 : 터미널창에 'del' 입력 후 enter   
- 프로그램 종료    : 터미널창에 'quit' 입력 후 enter  
- 카메라 모드 open : python dx_realtime_demo.py --features_path 0   

### Video Demo (batch input)
```bash
python dx_video_demo.py
```
### dxnn file porting
please using python 3.11 version,    
```bash
conda activate pia-package-executor
cd dx_rt
./build.sh
cd python_package
pip uninstall dx_engine
pip install .
```
Make sure there is a file *_pydxrt.cpython-311-x86_64-linux-gnu.so* under folder *dx_rt/python_package/src/dx_engine/capi*    
### Example
```python
from dx_engine import InferenceEngine
ie = InferenceEngine(your_model_path)
.
.
.
output = ie.run(input)
```
## 0. Prepare Linux
| Ubuntu 18.04 | Ubuntu 20.04  | Ubuntu 22.04  | Ubuntu 24.04 |
|--------------|---------------|---------------|--------------|
| Not Tested   | Not Tested    | Tested        | Not Tested   |

## 1. Set up Virtual Environment
```bash
conda create -n pia-package-executor python=3.11
conda activate pia-package-executor
```
## 2. Install Python Packages
```bash
pip install -r requirements.txt
```
## 3. Install PIA Space Packages
3.1. Download the following PIA AI packages.
- [pia-1.3.1obf-py-none-any.whl](https://bitbucket.org/pia-space/pia-ai-package/downloads/pia-1.3.1obf-py3-none-any.whl)
- [sub_clip4clip-1.2.3obf-py-none-any.whl](https://bitbucket.org/pia-space/sub-clip4clip/downloads/sub_clip4clip-1.2.3obf-py3-none-any.whl)

3.2. Place the downloaded wheel files into the project directory.
```
PROJECT_ROOT/
    +- assets/
    +- logs/
    +- README.md
    +- ...
    +- pia-1.3.1obf-py3-none-any.whl  # <- Place it into the project directory
    +- sub_clip4clip-1.2.3obf-py-none-any.whl  # <- Place it into the project directory
```

3.3. Install the PIA Space AI packages.
```bash
pip install pia-1.3.1obf-py3-none-any.whl
pip install sub_clip4clip-1.2.3obf-py3-none-any.whl
```

## 4. Install `onnxruntime`
4.2. Install the `onnxruntime` 
4.2.1.
For CPU:
```
pip install onnxruntime
```
# Experimental Assets Setup
Prepare the models and dataset following [this](assets/README.md).

# How to Run Evaluation
With small test set (10 pairs of videos and captions):
```bash
bash eval_sampled_test_set.sh
```
With full test set (1,000 pairs of videos and captions):
```bash
bash eval_full_test_set.sh
```
The similarity matrix and metrics results are saved to `outputs/sim_matrix.xlsx` and `outputs/metrics.xlsx`, respectively.