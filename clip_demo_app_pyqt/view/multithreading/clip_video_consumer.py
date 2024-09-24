import logging
import time
import traceback

import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from PyQt5.QtCore import pyqtSignal
from overrides import overrides

from clip_demo_app_pyqt.common.parser.parser_util import ParserUtil
from clip_demo_app_pyqt.lib.clip.dx_video_encoder import DXVideoEncoder
from clip_demo_app_pyqt.viewmodel.clip_view_model import ClipViewModel
from clip_demo_app_pyqt.view.multithreading.video_consumer import VideoConsumer


class ClipVideoConsumer(VideoConsumer):
    __dxnn_video_encoder = DXVideoEncoder(ParserUtil.get_args().video_encoder_dxnn)

    __clear_text_output_signal = pyqtSignal(int)
    __update_text_output_signal = pyqtSignal(int, str, int)

    # TODO : 불필요 여부 확인 필요
    # render_text_list_signal = pyqtSignal()

    def __init__(self, channel_idx: int, number_of_alarms: list, origin_video_frame_updated_signal: pyqtSignal,
                 video_source_changed_signal: pyqtSignal, sentence_list_update_signal: pyqtSignal, ctx: ClipViewModel):
        super().__init__(channel_idx, origin_video_frame_updated_signal, video_source_changed_signal)
        self.ctx = ctx

        self.__number_of_alarms = number_of_alarms
        self.__np_array_similarity = None
        self.__init_np_array_similarity()
        self.__last_update_time_text = 0  # Initialize the last update time
        self.__interval_update_time_text = 1.0  # Set update interval to 1 seconds (adjust as needed)
        self.__frame_count = 0
        self.__dxnn_fps = -1.0
        self.__sol_fps = -1.0

        self.image_transform = self.__transform(224)
        self.video_mask = torch.ones(1, 1)

        sentence_list_update_signal.connect(self.__init_np_array_similarity)

    @overrides()
    def process(self, frame):
        if frame is None:
            print("QImage is None on VideoConsumer" + str(frame))
            return
        # print("get QImage on VideoConsumer" + str(frame))

        # for prevent UI freeze
        time.sleep(0.3)

    # video_thread_list = self.ctx.get_video_thread_list()
    # for index in range(len(video_thread_list)):
        s = time.perf_counter_ns()

        similarity_list = []
        # vCap = video_thread_list[index]

        # TODO : 동영상 변경되었을때 감지하여 초기화 추가 필요 : self.cleanup()
        # if vCap.status_video_source():
        #     vCap.similarity_list = np.zeros((len(self.ctx.get_sentence_list())))
        #     vCap.__last_update_time_text = 0
        #     __frame_count = 0

        # frame = vCap.get_current_video_frame.copy()
        # TODO : copy() 필요할지 확인 필요
        dxnn_s = time.perf_counter_ns()
        input_data = self.image_transform(Image.fromarray(frame).convert("RGB"))
        video_pred = self.__dxnn_video_encoder.run(input_data)[0]
        dxnn_e = time.perf_counter_ns()
        # print(index, " : ", video_pred.shape)

        sentence_vector_list = self.ctx.get_sentence_vector_list()
        sentence_list = self.ctx.get_sentence_list()
        sentence_alarm_threshold_list = self.ctx.get_sentence_alarm_threshold_list()

        for text_index in range(len(sentence_vector_list)):
            try:
                ret = self.__loose_similarity(sentence_vector_list[text_index], video_pred, self.video_mask)
            except Exception as ex:
                # traceback.print_exc()
                logging.debug(ex)
                return
            similarity_list.append(ret)

        try:
            if len(similarity_list) > 0:
                np_array_similarity = np.stack(similarity_list).reshape(len(sentence_vector_list))
                if np_array_similarity.shape == self.__np_array_similarity.shape:
                    self.__np_array_similarity += np_array_similarity
                else:
                    return

        except Exception as ex:
            # traceback.print_exc()
            logging.debug(ex)
            return

        e = time.perf_counter_ns()
        dxnn_fps = 1000 / ((dxnn_e - dxnn_s) / 1000000)
        sol_fps = 1000 / ((e - s) / 1000000)
        self.__dxnn_fps = dxnn_fps
        self.__sol_fps = sol_fps

        self._update_each_fps(dxnn_fps, sol_fps)

    # for index in range(len(video_thread_list)):

        # vCap = video_thread_list[index]
        self.__update_argmax_text(sentence_list, self.__np_array_similarity / (self.__frame_count + 1),
                                  sentence_alarm_threshold_list)

        if self._channel_idx == 0:
            self._update_overall_fps()
        self.__frame_count += 1

    # TODO : 불필요 여부 확인 필요
    # def refresh_sentence_list(self):
    #     self.render_text_list_signal.emit()

    def get_update_text_output_signal(self):
        return self.__update_text_output_signal

    def get_clear_text_output_signal(self):
        return self.__clear_text_output_signal

    # TODO : 불필요 여부 확인 필요
    # def get_render_text_list_signal(self):
    #     return self.__render_text_list_signal

    @staticmethod
    def __transform(n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __update_argmax_text(self, text_list, logit_list, alarm_list):
        current_update_time_text = time.time()
        if current_update_time_text - self.__last_update_time_text < self.__interval_update_time_text:
            return

        argmax_info_list = []
        sorted_index = np.argsort(logit_list)
        indices_index = np.array(sorted(sorted_index[(-1 * self.__number_of_alarms):]))
        for t in indices_index:
            try:
                value = logit_list[t]
                min_value = alarm_list[t][0]
                max_value = alarm_list[t][1]
                alarm_threshold = alarm_list[t][2]
                if value < min_value:
                    ret_level = 0
                elif value > max_value:
                    ret_level = 100
                else:
                    ret_level = int((value - min_value) / (max_value - min_value) * 100)
                if value > alarm_threshold:
                    # print(value, ", ", alarm_threshold)
                    argmax_info_list.append({"text": text_list[t], "percent": ret_level})
            except Exception as ex:
                # traceback.print_exc()
                logging.debug(ex)
                continue

        self.__clear_text_output_signal.emit(self._channel_idx)
        for argmax_info in argmax_info_list:
            self.__update_text_output_signal.emit(self._channel_idx, argmax_info["text"], argmax_info["percent"])

        self.__last_update_time_text = current_update_time_text

    @overrides()
    def _cleanup(self):
        self.__init_np_array_similarity()
        self.__last_update_time_text = 0  # Initialize the last update time
        self.__interval_update_time_text = 1  # Set update interval to 1 seconds (adjust as needed)
        self.__frame_count = 0
        self.__dxnn_fps = -1.0
        self.__sol_fps = -1.0

    @staticmethod
    def __mean_pooling_for_similarity_visual(vis_output, video_frame_mask):
        video_mask_un = video_frame_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = vis_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.0] = 1.0
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def __loose_similarity(self, text_vectors, video_vectors, video_frame_mask):
        sequence_output, visual_output = (
            text_vectors.contiguous(),
            video_vectors.contiguous(),
        )
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self.__mean_pooling_for_similarity_visual(
            visual_output, video_frame_mask
        )
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        try:
            if sequence_output.ndim > 1:
                sequence_output = sequence_output.squeeze(1)
        except Exception as ex:
            # traceback.print_exc()
            logging.debug(ex)
            return
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        retrieve_logits = torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def __init_np_array_similarity(self):
        self.__np_array_similarity = np.zeros((len(self.ctx.get_sentence_list())))
