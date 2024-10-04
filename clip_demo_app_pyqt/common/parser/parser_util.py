import argparse

class ParserUtil:
    @staticmethod
    def get_args():
        # fmt: off
        parser = argparse.ArgumentParser(description="Generate Similarity Matrix from ONNX files")

        # Dataset Path Arguments
        parser.add_argument("--base_path", type=str, default="assets", help="Videos directory")

        # Dataset Configuration Arguments
        parser.add_argument("--max_words", type=int, default=32, help="")
        parser.add_argument("--feature_framerate", type=int, default=1, help="")
        parser.add_argument("--slice_framepos", type=int, default=0, choices=[0, 1, 2],
                            help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")

        # App Configuration Argument
        parser.add_argument("--number_of_channels", type=int, default=16, help="Number of input video channels")
        parser.add_argument("--terminal_mode", type=int, default=0, help="Terminal Mode Setting (off: 0, on: 1)")
        parser.add_argument("--camera_mode", type=int, default=0, help="Camera Mode Setting (off: 0, on: 1)")
        parser.add_argument("--blocking_mode", type=int, default=0, help="Blocking Mode Setting (off: 0, on: 1)")

        # Model Path Arguments
        parser.add_argument("--token_embedder_onnx", type=str,
                            default="assets/onnx/embedding_f32_op14_clip4clip_msrvtt_b128_ep5.onnx",
                            help="ONNX file path for token embedder")
        parser.add_argument("--text_encoder_onnx", type=str,
                            default="assets/onnx/textual_f32_op14_clip4clip_msrvtt_b128_ep5.onnx",
                            help="ONNX file path for text encoder")
        parser.add_argument("--video_encoder_onnx", type=str,
                            default="assets/onnx/visual_f32_op14_clip4clip_msrvtt_b128_ep5.onnx",
                            help="ONNX file path for video encoder")
        parser.add_argument("--video_encoder_dxnn", type=str, default="assets/dxnn/pia_vit_240814.dxnn",
                            help="ONNX file path for video encoder")

        return parser.parse_args()