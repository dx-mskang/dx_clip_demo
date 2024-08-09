# Environment Setup

## PIA DEMO (Draft)
### Real Time Demo (Average of outputs)
```bash
python dx_realtime_demo.py
```
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

## 4. Install `onnxruntime-gpu`
The way to install the `onnxruntim-gpu` depends on the CUDA version(https://onnxruntime.ai/docs/install/)

4.1. Check your CUDA version.
```bash
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0

```

4.2. Install the `onnxruntime-gpu` depending on your CUDA version.

For `CUDA-11.x`:
```
pip install onnxtuntime-gpu
```

For `CUDA-12.x`:
```
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
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