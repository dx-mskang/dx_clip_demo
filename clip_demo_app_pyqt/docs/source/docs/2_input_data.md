# Sample Input Data 

## Video Input

The demo app is configured to handle 16 videos located in `assets/demo_videos/*.mp4`, as defined in `clip_demo_app_pyqt/data/input_data.py` as follows.

```python
video_path_lists: list = [
    [
        "demo_videos/fire_on_car",
    ],
    [
        "demo_videos/dam_explosion_short",
    ],
    ...,
]
```

## Text Input

Each video is associated with 26 sentences, along with the corresponding `score min`, `score max`, and `score threshold` values, which are configured in `clip_demo_app_pyqt/data/input_data.py` as follows.
   
```python
sentence_list: list[Sentence] = [
    # video : "demo_videos/crowded_in_subway",
    Sentence("The subway is crowded with people",
        0.27, 0.29, 0.28),
    Sentence("People is crowded in the subway",
        0.27, 0.29, 0.28),

    # video : "demo_videos/heavy_structure_falling",
    Sentence("Heavy objects are fallen",
        0.21, 0.25, 0.225),
    ...,
```
