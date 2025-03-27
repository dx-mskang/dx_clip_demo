# Environment Setup

## CLIP DEMO using python-opencv

---
### Pre-Requisite
#### Get assets (input videos and prebuilt CLIP AI model) files
Download the asset files using setup.sh. These files are provided separately and are not included in the distributed source code.

When running setup.sh, you must specify app_type and dxrt_src_path as arguments.
Running setup.sh will automatically download the asset files and set up a Python virtual environment (venv).
`./setup.sh --app_type=<app_type> --dxrt_src_path=<path_to_dxrt>`
```bash
# exam
./setup.sh --app_type=opencv --dxrt_src_path=/deepx/dx_rt
```

File structure in `./assets/`:
```
assets
├── CLIP
│   ├── build
│   ├── clip
│   ├── clip.egg-info
│   ├── CLIP.png
│   ├── data
│   ├── hubconf.py
│   ├── LICENSE
│   ├── MANIFEST.in
│   ├── model-card.md
│   ├── notebooks
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.py
│   └── tests
├── data
│   ├── MSRVTT_JSFUSION_test.csv
│   └── MSRVTT_Videos
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
│   ├── clip_vit_240912.dxnn
└── onnx
    ├── embedding_f32_op14_clip4clip_msrvtt_b128_ep5.onnx
    ├── textual_f32_op14_clip4clip_msrvtt_b128_ep5.onnx
    └── visual_f32_op14_clip4clip_msrvtt_b128_ep5.onnx
```

---
### Setup Demo
#### 1. Run setup script 
Depending on the application runtime environment, an automated setup script can be used as shown below. For more details, please refer to the script's specifics.

Using setup.sh, the script automatically detects the amd64 and aarch64 environments, downloads the assets files, and sets up the application runtime environment.

However, for Windows, you need to run either setup_clip_demo_app_pyqt.bat or setup_clip_demo_app_opencv.bat.

##### for Linux amd64
```
./scripts/setup_clip_demo_app.sh --app_type=opencv --arch_type=amd64 --dxrt_src_path=<path_to_dxrt>
```

##### for Linux aarch64 (OrangePi 5+ Ubuntu 22.04 OS)
- for Linux aarch64 
```
./scripts/setup_clip_demo_app.sh --app_type=opencv --arch_type=aarch64 --dxrt_src_path=<path_to_dxrt>
```

##### for Windows
```
.\scripts\x86_64_win\setup_clip_demo_app_opencv.bat
```

### Execute Demo
#### 1. Activate python virtual environments
##### for Linux
```bash
source venv-opencv/bin/activate
```
##### for Windows
```
venv-opencv\bin\activate.bat
```

##### 2. Run Real Time Multi Channel Demo (16 channel - Average of outputs)
```bash
python clip_demo_app_opencv/dx_realtime_multi_demo.py
```
(P.S) To exit the demo application, simply press the `q` key. 

##### 3. Real Time Demo (Average of outputs)
```bash
python clip_demo_app_opencv/dx_realtime_demo.py
```
- Add a text sentence: Type the sentence in the terminal and press Enter
- Delete the last sentence: Type 'del' in the terminal and press Enter
- Exit the program: Type 'quit' in the terminal and press Enter
- Open camera mode: Run python dx_realtime_demo.py` --features_path 0`

##### 4. Video Demo (batch input)
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
