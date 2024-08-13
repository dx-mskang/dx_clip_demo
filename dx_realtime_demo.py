import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from pia.model import PiaONNXTensorRTModel
from sub_clip4clip.modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from tqdm import tqdm

import cv2
import threading

from dx_engine import InferenceEngine

# for demo 
import curses

input_text = ""
global_quit = False

def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate Similarity Matrix from ONNX files")
    
    # Dataset Path Arguments
    parser.add_argument("--val_csv", type=str, default="assets/data/sampled/MSRVTT_JSFUSION_test_10.csv", help="CSV file path of caption labels")
    # parser.add_argument("--features_path", type=str, default="assets/data/full/MSRVTT_Videos", help="Videos directory")
    parser.add_argument("--features_path", type=str, default="assets/demo_videos", help="Videos directory")
    
    # Dataset Configuration Arguments
    parser.add_argument("--max_words", type=int, default=20, help="")
    parser.add_argument("--feature_framerate", type=int, default=1, help="")
    parser.add_argument("--max_frames", type=int, default=100, help="")
    parser.add_argument("--eval_frame_order", type=int, default=0, choices=[0, 1, 2], help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument("--slice_framepos", type=int, default=0, choices=[0, 1, 2], help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    
    # Model Path Arguments
    parser.add_argument("--token_embedder_onnx", type=str, default="assets/onnx/embedding_f32_op14_clip4clip_msrvtt_b128_ep5.onnx", help="ONNX file path for token embedder")
    parser.add_argument("--text_encoder_onnx", type=str, default="assets/onnx/textual_f32_op14_clip4clip_msrvtt_b128_ep5.onnx", help="ONNX file path for text encoder")
    parser.add_argument("--video_encoder_onnx", type=str, default="assets/onnx/visual_f32_op14_clip4clip_msrvtt_b128_ep5.onnx", help="ONNX file path for video encoder")
    parser.add_argument("--torch_model", type=str, default="assets/pth/clip4clip_msrvtt_b128_ep5.pth", help="pth file path for torch model")
    
    return parser.parse_args()
    # fmt: on


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif sys.platform.startswith("darwin"):
        return "mps"
    else:
        return "cpu"

def insert_text_in_term():
    global input_text
    global global_quit
    while True:
        input_text = input("Enter text to display on video (insert 'quit' to quit): ")
        if input_text == "quit":
            global_quit = True
            break

def _mean_pooling_for_similarity_visual(vis_output, video_frame_mask):
    video_mask_un = video_frame_mask.to(dtype=torch.float).unsqueeze(-1)
    visual_output = vis_output * video_mask_un
    video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
    video_mask_un_sum[video_mask_un_sum == 0.0] = 1.0
    video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
    return video_out


def _loose_similarity(text_vectors, video_vectors, video_frame_mask):
    sequence_output, visual_output = (
            text_vectors.contiguous(),
            video_vectors.contiguous(),
        )
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    visual_output = _mean_pooling_for_similarity_visual(
        visual_output, video_frame_mask
    )
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

    sequence_output = sequence_output.squeeze(1)
    sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
    retrieve_logits = torch.matmul(sequence_output, visual_output.t())
    return retrieve_logits

class DXVideoEncoder():
    def __init__(self, model_path: str):
        self.ie = InferenceEngine(model_path)
        
    def run(self, x):
        x = self.preprocess_torch(x).numpy()
        output = self.ie.run(x)
        y = output[0]
        z = self.postprocess_torch(torch.from_numpy(y))
        return z
        
        
        
    def preprocess_numpy(
        self,
        x: np.ndarray,
        mul_val: np.ndarray = np.float32([64.75]),
        add_val: np.ndarray = np.float32([-11.949951171875]),
        ) -> np.ndarray:
        x = x.astype(np.float32)
        x = x * mul_val + add_val
        x = x.round().clip(-128, 127)
        x = x.astype(np.int8)
        x = np.reshape(x, [1, 3, 7, 32, 7, 32])
        x = np.transpose(x, [0, 2, 4, 3, 5, 1])
        x = np.reshape(x, [1, 49, 48, 64])
        x = np.transpose(x, [0, 2, 1, 3])
        return x

    def preprocess_torch(
            self,
            x: torch.Tensor,
            mul_val: torch.Tensor = torch.FloatTensor([64.75]),
            add_val: torch.Tensor = torch.FloatTensor([-11.949951171875]),
        ) -> torch.Tensor:
        x = x.to(torch.float32)
        x = x * mul_val + add_val
        x = x.round().clip(-128, 127)
        x = x.to(torch.int8)
        x = torch.reshape(x, [1, 3, 7, 32, 7, 32])
        x = torch.permute(x, [0, 2, 4, 3, 5, 1])
        x = torch.reshape(x, [1, 49, 48, 64])
        x = torch.permute(x, [0, 2, 1, 3])
        return x

    def postprocess_numpy(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 3
        x = x[:, 0]
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def postprocess_torch(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3
        x = x[:, 0]
        return x / torch.norm(x, dim=-1, keepdim=True)

class VideoThread(threading.Thread):
    def __init__(self, features_path, video_paths, gt_text_list):
        super().__init__()
        self.features_path = features_path
        self.video_paths = video_paths
        self.gt_text_list = gt_text_list
        self.current_index = 0
        self.video_path_current = os.path.join(features_path, video_paths[self.current_index] + ".mp4")
        self.cap = cv2.VideoCapture(self.video_path_current)
        self.stop_thread = False
        self.final_text = ""
        self.result_text = ""
        self.result_logit = 0.0
        self.released = False
        self.pannel_size_w = 700 
        self.pannel_size_h = 700
        self.video_size_w = 640
        self.video_size_h = 480
        self.text_pannel_frame = np.ones((self.pannel_size_w, self.pannel_size_h, 3), dtype=np.uint8) * 255
        self.new_text_pannel_frame = np.ones((self.pannel_size_w, self.pannel_size_h, 3), dtype=np.uint8) * 255
        self.original = np.zeros((224, 224, 3), dtype=np.uint8)
        self.transform_ = self.transform(224)
        self.update_new_text = False
        self.pop_last_text = False
        self.new_text = ""

    def run(self):
        global input_text
        global global_quit
        i = 0
        for text_i in self.gt_text_list:
            cv2.putText(self.text_pannel_frame, "{}. ".format(i) + text_i, (5, 15 + (20 * i)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            i+=1
        self.new_text_pannel_frame = self.text_pannel_frame.copy()
        while not self.stop_thread:
            ret, self.original = self.cap.read()
            if global_quit:
                self.released = True
                break 
            if not ret:
                # 현재 비디오가 끝났을 때 다음 비디오로 넘어감
                self.current_index += 1
                if self.current_index < len(self.video_paths):
                    self.video_path_current = os.path.join(self.features_path, self.video_paths[self.current_index] + ".mp4")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_path_current)
                    continue
                else:
                    self.current_index= 0
                    self.video_path_current = os.path.join(self.features_path, self.video_paths[self.current_index] + ".mp4")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_path_current)
                    continue
            
            # 영상 위에 텍스트 추가
            frame = cv2.resize(self.original, (self.video_size_w, self.video_size_h), cv2.INTER_NEAREST)
            cv2.rectangle(frame, (0, self.video_size_h-70), (self.video_size_w, self.video_size_h-40), (0, 0, 0), -1)
            cv2.putText(frame, self.final_text, (10, self.video_size_h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            if self.update_new_text:
                cv2.putText(self.text_pannel_frame, "{}. ".format(i) + self.new_text, (5, 15 + (20 * i)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                i+=1
                self.update_new_text = False
            
            if self.pop_last_text:
                i-=1
                cv2.rectangle(self.text_pannel_frame, (0, 15 + (20 * i) - 15), (self.pannel_size_w, 15 + (20 * i) + 15), (255, 255, 255), -1)
                self.new_text_pannel_frame = self.text_pannel_frame.copy()
                self.pop_last_text = False
                self.final_text = " "
            cv2.namedWindow('Text')  
            cv2.namedWindow('Video')  
            cv2.moveWindow('Text', self.video_size_w, 0)
            cv2.moveWindow('Video', 0, 0)
            cv2.imshow('Text', self.new_text_pannel_frame)
            cv2.imshow('Video', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
        self.released = True
        self.cap.release()
        cv2.destroyAllWindows()
    
    def stop(self):
        self.stop_thread = True

    def update_text(self, argmax_index, new_text, new_logit):
        self.final_text = new_text + ",  sim : {:.3}".format(new_logit[0][0])
        self.new_text_pannel_frame = self.text_pannel_frame.copy()
        cv2.putText(self.new_text_pannel_frame, 
                    "{}. ".format(argmax_index) + self.gt_text_list[argmax_index] 
                    + ", sim : {:.3}".format(new_logit[0][0]), 
                    (5, 15 + (20 * argmax_index)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    
    def transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def get_input_tensor(self):
        try:
            frame_data = self.transform_(Image.fromarray(self.original).convert("RGB"))
        except Exception as e:
            frame_data = np.zeros((224, 224, 3), dtype=np.uint8)
            frame_data = np.transpose(frame_data, (2, 0, 1))
        input_data = torch.as_tensor(frame_data).float()
        input_data = input_data.unsqueeze(0)
        return input_data
    
    def update_text_vector(self, new_text_vector):
        self.new_text = new_text_vector
        self.update_new_text = True
    
    def pop_text_vector(self, index):
        self.pop_last_text = True


def main():
    global input_text
    
    SPECIAL_TOKEN = {
        "CLS_TOKEN": "<|startoftext|>",
        "SEP_TOKEN": "<|endoftext|>",
        "MASK_TOKEN": "[MASK]",
        "UNK_TOKEN": "[UNK]",
        "PAD_TOKEN": "[PAD]",
    }
    max_words = 32
    # Get Input Arguments
    args = get_args()

    # Get Device
    device = get_device()
    
    model_load_time_s = time.perf_counter_ns()
    
    # Set up ONNX Models (Used for Inference)
    video_encoder = PiaONNXTensorRTModel(
        model_path=args.video_encoder_onnx, device=device
    )
    
    dxnn_video_encoder = DXVideoEncoder("dxnn/pia_vit/pia_vit_240812.dxnn")
    
    token_embedder = PiaONNXTensorRTModel(
        model_path=args.token_embedder_onnx, device=device
    )
    text_encoder = PiaONNXTensorRTModel(
        model_path=args.text_encoder_onnx, device=device
    )
    
    model_load_time_e = time.perf_counter_ns()
    print("[TIME] Model Load : {} ns".format(model_load_time_e - model_load_time_s))
    
    gt_video_path_list = [
        "video9770", "video9771", "video7020", "video9773", "video7026", 
        "video9775", "video9776", "video9778", "video9779", 
        "video7028", "video7029", "video9772", "video7021", "video9774", 
        "video7027", "video9731", "video7024", "video9777", "video8913", 
        "video8912", "video8910", "video8917", "video8916", 
        "video8915", "video8914", "video8919", "video8918", "video9545"
    ]
    
    # gt_video_path_list = [
    #         "0",
    #         "1",
    #         "2",
    #         "3",
    #         "4",
    #         "5",
    #         "6",
    #         "7"
    # ]
    
    gt_text_list = [ 
        "a person is connecting something to system", 
        "a little girl does gymnastics", 
        "a woman creating a fondant baby and flower", 
        "a boy plays grand theft auto 5", 
        "a man is giving a review on a vehicle", 
        "a man speaks to children in a classroom", 
        "one micky mouse is talking to other", 
        "a naked child runs through a field", 
        "a little boy singing in front of judges and crowd", 
        "fireworks are being lit and exploding in a night sky", 
        "a man is singing and standing in the road", 
        "cartoon show for kids", 
        "some cartoon characters are moving around an area", 
        "baseball player hits ball", 
        "a rocket is lauching into a blue sky smoke is emerging from the base of the rocket", 
        "the man in the video is showing a brief viewing of how the movie is starting", 
        "a woman is mixing food in a mixing bowl", 
        "little pet shop cat getting a bath and washed with little brush", 
        "a student explains to his teacher about the sheep of another student", 
        "a video about different sports", 
        "a family is having coversation", 
        "adding ingredients to a pizza", 
        "two men discuss social issues", 
        "cartoons of a sponge a squid and a starfish", 
        "person cooking up somefood", 
        "models are walking down a short runway", 
        "a man is talking on stage", 
        "a hairdresser and client speak to each other with kid voices", 
        "some one talking about top ten movies of the year"
    ]
    
    # gt_text_list = [ 
    #         "sports people are fighting on field"
    # ]
    
    video_thread = VideoThread(args.features_path, gt_video_path_list, gt_text_list)
    
    # text embedding vector 미리 준비 
    text_vector_list = []
    
    # 시간 측정
    t_text_tokenizer_list = []
    t_text_token_to_id = []
    t_text_embedding = []
    t_text_encoding = []
    for i in tqdm(range(len(gt_text_list))):
        gt_text_sample = gt_text_list[i]
        
        tokenizer_time_s = time.perf_counter_ns()
        # Get Token,
        # ex ) "some one talking about top ten movies of the year" 
        #      -> [some, one, talking, about, top, ten, movies, of, the, year]
        gt_text_token = ClipTokenizer().tokenize(gt_text_sample)
        
        tokenizer_time_e = time.perf_counter_ns()
        t_text_tokenizer_list.append(tokenizer_time_e - tokenizer_time_s)
        
        gt_text_token = [SPECIAL_TOKEN["CLS_TOKEN"]] + gt_text_token
        total_length_with_class = max_words - 1
        if len(gt_text_token) > total_length_with_class:
            gt_text_token = gt_text_token[:total_length_with_class]
        gt_text_token = gt_text_token + [SPECIAL_TOKEN["SEP_TOKEN"]]
        
        token_to_id_time_s = time.perf_counter_ns()
        
        # 9 text ids : number of token ids
        # token to ids (using by "bpe_simple_vocab_16e6.txt.gz")
        raw_text_data = ClipTokenizer().convert_tokens_to_ids(gt_text_token)
        token_to_id_time_e = time.perf_counter_ns()
        t_text_token_to_id.append(token_to_id_time_e - token_to_id_time_s)
        
        text_input_mask = [1] * len(raw_text_data) + [0] * (
                    max_words - len(raw_text_data)
                )

        text_input_ids = raw_text_data + [0] * (
                    max_words - len(raw_text_data)
                )
        
        text_input_mask = torch.tensor(
                        [text_input_mask], 
                        ).to(device, dtype=torch.float32)
        
        text_input_ids = torch.tensor(
                        [text_input_ids], 
                        ).to(device)
        
        
        text_enmbedding_time_s = time.perf_counter_ns()
        # [1, 32, 512]
        text_embedding = token_embedder(text_input_ids)    
        text_enmbedding_time_e = time.perf_counter_ns()
        t_text_embedding.append(text_enmbedding_time_e - text_enmbedding_time_s)
        
        text_encoder_time_s = time.perf_counter_ns()
        # [1, 512]
        text_vectors = text_encoder([text_embedding, text_input_mask])
        text_encoder_time_e = time.perf_counter_ns()
        t_text_encoding.append(text_encoder_time_e - text_encoder_time_s)

        text_vector_list.append(text_vectors)
    
    # 시간 측정
    t_video_get_data = []
    t_video_encoder = []
    t_sim_value = []
    
    video_thread.start()
    video_path = video_thread.current_index
    j = 0
    result_logits_np = np.zeros((len(text_vector_list), 1, 1))
    print("Enter text to display on video (insert 'quit' to quit): ")
    while True:
        raw_video_data = video_thread.get_input_tensor()
        raw_video_mask_data = torch.ones(1, raw_video_data.shape[0])
        
        video_pred = video_encoder(raw_video_data)
        # video_pred = dxnn_video_encoder.run(raw_video_data)
        
        if False:
            video_pred = torch.gather(video_pred, 1, torch.from_numpy(np.zeros((1, 512), dtype=np.int64)))
            reduceL2_output = torch.norm(video_pred, keepdim=True)
            video_pred = video_pred/reduceL2_output
        result_logits = []
        for k in range(len(text_vector_list)):
            retrieve_logits = _loose_similarity(text_vector_list[k], video_pred, raw_video_mask_data)
            result_logits.append(retrieve_logits)
            
        if len(text_vector_list) > 0 :
            result_logits_np += np.stack(result_logits)
            
            argmax_index = np.argmax(result_logits_np / (j+1))
            video_thread.update_text(argmax_index,gt_text_list[argmax_index], result_logits_np[argmax_index]/(j+1))
        # print("max retrieve_logits = {}, argmax = {}".format(result_logits_np[argmax_index]/(j+1), argmax_index))
        if input_text == "del":
            j = 0
            if len(text_vector_list) > 0:
                text_vector_list.pop(-1)
                result_logits_np = np.zeros((len(text_vector_list), 1, 1))
                gt_text_list.pop(-1)
                video_thread.pop_text_vector(-1)
            input_text = ""
        elif input_text != "":
            gt_text_token = ClipTokenizer().tokenize(input_text)
            gt_text_token = [SPECIAL_TOKEN["CLS_TOKEN"]] + gt_text_token
            total_length_with_class = max_words - 1
            if len(gt_text_token) > total_length_with_class:
                gt_text_token = gt_text_token[:total_length_with_class]
            gt_text_token = gt_text_token + [SPECIAL_TOKEN["SEP_TOKEN"]]
            raw_text_data = ClipTokenizer().convert_tokens_to_ids(gt_text_token)
            text_input_mask = [1] * len(raw_text_data) + [0] * (
                        max_words - len(raw_text_data)
                    )

            text_input_ids = raw_text_data + [0] * (
                        max_words - len(raw_text_data)
                    )
            
            text_input_mask = torch.tensor(
                            [text_input_mask], 
                            ).to(device, dtype=torch.float32)
            
            text_input_ids = torch.tensor(
                            [text_input_ids], 
                            ).to(device)
            text_embedding = token_embedder(text_input_ids)    
            text_vectors = text_encoder([text_embedding, text_input_mask])
            text_vector_list.append(text_vectors)
            j = 0
            result_logits_np = np.zeros((len(text_vector_list), 1, 1))
            video_thread.update_text_vector(input_text)
            gt_text_list.append(input_text)
            input_text = ""
        j += 1
        if(video_thread.current_index != video_path):
            j = 0
            if len(text_vector_list) > 0 :
                result_logits_np = np.zeros((len(text_vector_list), 1, 1))
        if video_thread.released:
            break
    
    # # print Profile
    # avg_t_text_tokenizer_list = np.average(np.stack(t_text_tokenizer_list))
    # avg_t_text_token_to_id = np.average(np.stack(t_text_token_to_id))
    # avg_t_text_embedding = np.average(np.stack(t_text_embedding))
    # avg_t_text_encoding = np.average(np.stack(t_text_encoding))
    
    # avg_t_video_encoder = np.average(np.stack(t_video_encoder))
    # avg_t_sim_value = np.average(np.stack(t_sim_value))
    
    # print("\n")
    # print("** Average of text tokenizer = {} ns".format(avg_t_text_tokenizer_list))
    # print("** Average of text token to ids = {} ns".format(avg_t_text_token_to_id))
    # print("** Average of text embedding = {} ns".format(avg_t_text_embedding))
    # print("** Average of text encoding = {} ns".format(avg_t_text_encoding))
    # print("\n")
    
    # print("** Average of get video encoded data for 30 frames = {} ns".format(avg_t_video_encoder))
    # print("** Average of calculation similarity = {} ns".format(avg_t_sim_value))
    # print("\n")
    
if __name__ == "__main__":
    text_thread = threading.Thread(target=insert_text_in_term)
    text_thread.daemon = True
    text_thread.start()
    
    main()
    
    text_thread.join()
