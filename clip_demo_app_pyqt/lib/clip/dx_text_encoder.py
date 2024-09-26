import sys
import time

import torch
from pia.model import PiaONNXTensorRTModel
from sub_clip4clip.modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from tqdm import tqdm

from clip_demo_app_pyqt.common.parser.parser_util import ParserUtil


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

    token_embedder = PiaONNXTensorRTModel(
        model_path=ParserUtil.get_args().token_embedder_onnx, device=DEVICE
    )
    text_encoder = PiaONNXTensorRTModel(
        model_path=ParserUtil.get_args().text_encoder_onnx, device=DEVICE
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
        token = ClipTokenizer().tokenize(text)
        token = [cls.SPECIAL_TOKEN["CLS_TOKEN"]] + token
        total_length_with_class = cls.MAX_WORDS - 1
        if len(token) > total_length_with_class:
            token = token[:total_length_with_class]
        token = token + [cls.SPECIAL_TOKEN["SEP_TOKEN"]]
        token_ids = ClipTokenizer().convert_tokens_to_ids(token)
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
