#!/bin/bash
python jsj_eval.py \
    --val_csv assets/data/sampled/MSRVTT_JSFUSION_test_10.csv \
    --features_path assets/data/sampled/MSRVTT_Videos_10 \
    --token_embedder_onnx assets/onnx/embedding_f32_op14_clip4clip_msrvtt_b128_ep5.onnx \
    --text_encoder_onnx assets/onnx/textual_f32_op14_clip4clip_msrvtt_b128_ep5.onnx \
    --video_encoder_onnx dxnn/pia_vit_240814/pia_vit_240814.dxnn \
    --torch_model assets/pth/clip4clip_msrvtt_b128_ep5.pth