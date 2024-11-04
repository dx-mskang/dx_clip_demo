# Sample Input Data 

## Video Input

The demo app is configured to handle 16 videos located in `assets/demo_videos/*.mp4`. This configuration is managed in JSON format in the `clip_demo_app_pyqt/data/default_data.json` file.

```json
{
  "video_path_lists": [
    [
      "demo_videos/fire_on_car"
    ],
    [
      "demo_videos/dam_explosion_short"
    ],
    ...
  ]
}
```

## Text Input

Each video is associated with 26 sentences, along with the corresponding `score_min`, `score_max`, and `score_threshold` values. This data is also managed in JSON format in the `clip_demo_app_pyqt/data/default_data.json` file.

```json
{
  "sentence_list": [
    {
      "video": "demo_videos/crowded_in_subway",
      "sentences": [
        {
          "text": "The subway is crowded with people",
          "score_min": 0.27,
          "score_max": 0.29,
          "score_threshold": 0.28
        },
        {
          "text": "People are crowded in the subway",
          "score_min": 0.27,
          "score_max": 0.29,
          "score_threshold": 0.28
        }
      ]
    },
    {
      "video": "demo_videos/heavy_structure_falling",
      "sentences": [
        {
          "text": "Heavy objects are fallen",
          "score_min": 0.21,
          "score_max": 0.25,
          "score_threshold": 0.225
        }
      ]
    },
    ...
  ]
}
```
