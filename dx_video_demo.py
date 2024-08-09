import argparse
import os
import sys
import time

import numpy as np
import torch
from pia.model import PiaONNXTensorRTModel
from sub_clip4clip.modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from tqdm import tqdm

import cv2
import threading

def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate Similarity Matrix from ONNX files")
    
    # Dataset Path Arguments
    parser.add_argument("--val_csv", type=str, default="assets/data/sampled/MSRVTT_JSFUSION_test_10.csv", help="CSV file path of caption labels")
    parser.add_argument("--features_path", type=str, default="assets/data/full/MSRVTT_Videos", help="Videos directory")
    
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

class VideoThread(threading.Thread):
    def __init__(self, features_path, video_paths, gt_text_list):
        super().__init__()
        self.features_path = features_path
        self.video_paths = video_paths
        self.gt_text_list = gt_text_list
        self.current_index = 0
        self.video_path_current = os.path.join(features_path, video_paths[self.current_index] + ".mp4")
        self.cap = cv2.VideoCapture(self.video_path_current)
        frame_width = 640
        frame_height = 480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.stop_thread = False
        self.final_text = ""
        self.result_text = ""
        self.result_logit = 0.0
        self.released = False
        self.text_pannel_frame = np.ones((640, 960, 3), dtype=np.uint8) * 255
        self.new_text_pannel_frame = np.ones((640, 960, 3), dtype=np.uint8) * 255

    def run(self):
        for i, text_i in enumerate(self.gt_text_list):
            cv2.putText(self.text_pannel_frame, "{}. ".format(i) + text_i, (5, 15 + (20 * i)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        self.new_text_pannel_frame = self.text_pannel_frame.copy()
        while not self.stop_thread:
            ret, frame = self.cap.read()
            if not ret:
                # 현재 비디오가 끝났을 때 다음 비디오로 넘어감
                self.current_index += 1
                if self.current_index < len(self.video_paths):
                    self.video_path_current = os.path.join(self.features_path, self.video_paths[self.current_index] + ".mp4")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_path_current)
                    frame_width = 640
                    frame_height = 480
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
                    continue
                else:
                    self.released = True
                    break
            # 영상 위에 텍스트 추가
            frame = cv2.resize(frame, (640, 480), cv2.INTER_NEAREST)
            cv2.rectangle(frame, (0, 480-70), (640, 480-40), (0, 0, 0), -1)
            cv2.putText(frame, self.final_text, (10, 480-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Text', self.new_text_pannel_frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
    
    def stop(self):
        self.stop_thread = True

    def update_text(self, argmax_index, new_text, new_logit):
        self.final_text = new_text + ",  sim : {:.3}".format(new_logit[0][0])
        self.new_text_pannel_frame = self.text_pannel_frame.copy()
        cv2.putText(self.new_text_pannel_frame, "{}. ".format(argmax_index) + self.gt_text_list[argmax_index] + ", sim : {:.3}".format(new_logit[0][0]), (5, 15 + (20 * argmax_index)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def main():
    from sub_clip4clip.dataloaders.rawvideo_util import RawVideoExtractorCV2
    
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
    
    video_extractor_time_s = time.perf_counter_ns()
    
    video_extractor = RawVideoExtractorCV2(
        framerate=1.0, size=224
    )
    
    video_extractor_time_e = time.perf_counter_ns()
    
    model_load_time_s = time.perf_counter_ns()
    
    # Set up ONNX Models (Used for Inference)
    video_encoder = PiaONNXTensorRTModel(
        model_path=args.video_encoder_onnx, device=device
    )
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
        "video9775", "video9776", "video7025", "video9778", "video9779", 
        "video7028", "video7029", "video9772", "video7021", "video9774", 
        "video7027", "video9731", "video7024", "video9777", "video8913", 
        "video8912", "video8910", "video8917", "video8916", 
        "video8915", "video8914", "video8919", "video8918", "video9545"
    ]
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
    for j in range(len(gt_video_path_list)):
        file_name = gt_video_path_list[j]
        gt_video_path = os.path.join(args.features_path,file_name+".mp4")
        
        get_video_data_time_s = time.perf_counter_ns()
        # torch video tensor : 11, 3, 224, 224 (num windows, channel, height, width) 
        print(video_extractor.get_video_data(gt_video_path)['video'][0].shape)
        raw_video_data = video_extractor.get_video_data(gt_video_path)['video'][0].unsqueeze(0).to(device)
        if j == 0:
            video_thread.start()
        raw_video_mask_data = torch.ones(1, raw_video_data.shape[0])
        get_video_data_time_e = time.perf_counter_ns()
        t_video_get_data.append((get_video_data_time_e - get_video_data_time_s) / raw_video_data.shape[0])
        
        video_encoder_pred_time_s = time.perf_counter_ns()
        # [11, 512]
        video_pred = video_encoder(raw_video_data)
        print(video_pred.shape)
        video_encoder_pred_time_e = time.perf_counter_ns()
        t_video_encoder.append((video_encoder_pred_time_e - video_encoder_pred_time_s) / video_pred.shape[0])
        
        print("[{}] Calculate Video Data : ".format(j) + gt_video_path)
        result_logits = []
        for k in range(len(text_vector_list)):
            calculate_sim_time_s = time.perf_counter_ns()
            retrieve_logits = _loose_similarity(text_vector_list[k], video_pred, raw_video_mask_data)
            result_logits.append(retrieve_logits)
            calculate_sim_time_e = time.perf_counter_ns()
            t_sim_value.append(calculate_sim_time_e - calculate_sim_time_s)
        argmax_index = np.argmax(np.stack(result_logits))
        print("max retrieve_logits = {}, argmax = {}".format(result_logits[argmax_index], argmax_index))
        video_thread.update_text(argmax_index, gt_text_list[argmax_index], result_logits[argmax_index])
        
        while gt_video_path == video_thread.video_path_current:
            if gt_video_path != video_thread.video_path_current or video_thread.released:
                break
        if video_thread.released:
            break
        
    # print Profile
    avg_t_text_tokenizer_list = np.average(np.stack(t_text_tokenizer_list))
    avg_t_text_token_to_id = np.average(np.stack(t_text_token_to_id))
    avg_t_text_embedding = np.average(np.stack(t_text_embedding))
    avg_t_text_encoding = np.average(np.stack(t_text_encoding))
    
    avg_t_video_get_data = np.average(np.stack(t_video_get_data))
    avg_t_video_encoder = np.average(np.stack(t_video_encoder))
    avg_t_sim_value = np.average(np.stack(t_sim_value))
    
    print("\n")
    print("** Average of text tokenizer = {} ns".format(avg_t_text_tokenizer_list))
    print("** Average of text token to ids = {} ns".format(avg_t_text_token_to_id))
    print("** Average of text embedding = {} ns".format(avg_t_text_embedding))
    print("** Average of text encoding = {} ns".format(avg_t_text_encoding))
    print("\n")
    
    print("** Average of get video data for 30 frames = {} ns".format(avg_t_video_get_data))
    print("** Average of get video encoded data for 30 frames = {} ns".format(avg_t_video_encoder))
    print("** Average of calculation similarity = {} ns".format(avg_t_sim_value))
    print("\n")
    
if __name__ == "__main__":
    main()
