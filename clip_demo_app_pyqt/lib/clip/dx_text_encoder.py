import sys
import time

import torch
from clip.simple_tokenizer import SimpleTokenizer as ClipTokenizer
from tqdm import tqdm

from clip_demo_app_pyqt.common.parser.parser_util import ParserUtil
import onnxruntime
import os
class ONNXModel(torch.nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        assert os.path.isfile(model_path), f"can't load model file at {model_path}"
        self.model = onnxruntime.InferenceSession(model_path)
        self.output_names = [x.name for x in self.model.get_outputs()]
    
    def forward(self, x):
        if len(self.model.get_inputs()) != 1:
            inputs = {}
            for i in range(len(self.model.get_inputs())):
                if abs(len(self.model.get_inputs()[i].shape) - len(x[i].shape)) == 1 :
                    x[i] = x[i][0,...]
                inputs[self.model.get_inputs()[i].name] = x[i].cpu().numpy()
        else:
            x = x.cpu().numpy()
            inputs = {self.model.get_inputs()[0].name: x}
        pred = self.model.run(self.output_names, inputs)
        if isinstance(pred, list):
            import numpy as np
            pred = np.stack(pred)
        return torch.Tensor(pred, device="cpu")

class TextVectorUtil:
    SPECIAL_TOKEN = {
        "CLS_TOKEN": "<|startoftext|>",
        "SEP_TOKEN": "<|endoftext|>",
        "MASK_TOKEN": "[MASK]",
        "UNK_TOKEN": "[UNK]",
        "PAD_TOKEN": "[PAD]",
    }
    MAX_WORDS = 32
    DEVICE = ""
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif sys.platform.startswith("darwin"):
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    model_load_time_s = time.perf_counter_ns()

    token_embedder = ONNXModel(
        model_path=ParserUtil.get_args().token_embedder_onnx
    )
    text_encoder = ONNXModel(
        model_path=ParserUtil.get_args().text_encoder_onnx
    )

    model_load_time_e = time.perf_counter_ns()
    print("[TIME] Model Load : {} ns".format(model_load_time_e - model_load_time_s))

    @classmethod
    def get_text_vector_list(cls, text_list: list[str]):
        ret = []
        for i in tqdm(range(len(text_list))):
            text = text_list[i]
            text_vectors = cls.get_text_vector(text)
            ret.append(text_vectors)
        return ret

    @classmethod
    def get_text_vector(cls, text: str):
        text = text if len(text.split(" ")) < cls.MAX_WORDS - 1 else str.join(text.split(" ")[:cls.MAX_WORDS - 2])
        text = cls.SPECIAL_TOKEN["CLS_TOKEN"] + " " + text + " " + cls.SPECIAL_TOKEN["SEP_TOKEN"]
        token_ids = ClipTokenizer().encode(text)
        token_ids_mask = [1] * len(token_ids) + [0] * (
                cls.MAX_WORDS - len(token_ids)
        )
        token_ids = token_ids + [0] * (
                cls.MAX_WORDS - len(token_ids)
        )
        token_ids_mask = torch.tensor([token_ids_mask]).to(cls.DEVICE, dtype=torch.float32)
        token_ids = torch.tensor([token_ids]).to(cls.DEVICE)
        text_embedding = cls.token_embedder(token_ids)
        text_vectors = cls.text_encoder([text_embedding, token_ids_mask])
        return text_vectors
