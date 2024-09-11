import argparse
import sys
import os
import time
import math
from typing import List, Tuple
import threading
from pyexpat import features

import numpy as np
import torch
from PIL import Image
from overrides import overrides
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from pia.model import PiaONNXTensorRTModel
from sub_clip4clip.modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from tqdm import tqdm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit, QTextEdit, QVBoxLayout, QHBoxLayout,
    QGridLayout, QStackedLayout, QSpinBox, QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pyqttoast import Toast, ToastPreset, ToastPosition
import cv2
import qdarkstyle

# 필요한 모듈 임포트 및 클래스 정의
# ... (생략된 부분은 기존 코드에서 가져옵니다)
from dx_engine import InferenceEngine


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate Similarity Matrix from ONNX files")

    # Dataset Path Arguments
    parser.add_argument("--features_path", type=str, default="assets", help="Videos directory")

    # Dataset Configuration Arguments
    parser.add_argument("--number_of_channels", type=int, default=16, help="Number of input video channels")
    parser.add_argument("--max_words", type=int, default=32, help="")
    parser.add_argument("--feature_framerate", type=int, default=1, help="")
    parser.add_argument("--slice_framepos", type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")

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
    # fmt: on

class DXVideoEncoder:
    def __init__(self, model_path: str):
        self.ie = InferenceEngine(model_path)

    def run(self, x):
        x = x.numpy()
        x = self.preprocess_numpy(x)
        x = np.ascontiguousarray(x)
        o = self.ie.run(x)[0]
        o = self.postprocess_numpy(o)
        o = torch.from_numpy(o)
        return o

    @staticmethod
    def preprocess_numpy(
            x: np.ndarray,
            mul_val: np.ndarray = np.float32([64.75055694580078]),
            add_val: np.ndarray = np.float32([-11.950003623962402]),
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

    @staticmethod
    def preprocess_torch(
            x: torch.Tensor,
            mul_val: torch.Tensor = torch.FloatTensor([64.75055694580078]),
            add_val: torch.Tensor = torch.FloatTensor([-11.950003623962402]),
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

    @staticmethod
    def postprocess_numpy(x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 3
        x = x[:, 0]
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        return x

    @staticmethod
    def postprocess_torch(x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3
        x = x[:, 0]
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x


class SingleVideoThread(QThread):
    change_pixmap_signal = pyqtSignal(int, QImage)
    clear_text_output_signal = pyqtSignal(int)
    update_text_output_signal = pyqtSignal(int, str, int)
    render_text_list_signal = pyqtSignal()
    update_each_fps_signal = pyqtSignal(int, float, float)

    def __init__(self, base_path: str, video_path_list: List[str], thread_idx: int, video_size: Tuple, number_of_alarms):
        super().__init__()
        self._running = True
        # 초기화 코드
        # ...
        self.blocking_mode = True
        self.video_size = video_size
        self.video_fps = 30

        self.pause_thread = False
        self.number_of_alarms = number_of_alarms
        self.last_update_time_text = 0  # Initialize the last update time
        self.interval_update_time_text = 1  # Set update interval to 1 seconds (adjust as needed)
        self.last_update_time_fps = 0  # Initialize the last update time
        self.interval_update_time_fps = 0.3  # Set update interval to 0.3 seconds (adjust as needed)

        if video_path_list[0] == "/dev/video0":
            self.base_path = ""
            self.video_path_list = ["/dev/video0"]
            self.current_index = 0
            self.video_path_current = os.path.join(self.video_path_list[self.current_index])
        else:
            self.base_path = base_path
            self.video_path_list = video_path_list
            self.current_index = 0
            self.video_path_current = os.path.join(self.base_path, self.video_path_list[self.current_index] + ".mp4")

        self.cap = cv2.VideoCapture(self.video_path_current)
        self.current_original_frame = np.zeros((self.video_size[1], self.video_size[0], 3), dtype=np.uint8)
        self.thread_idx = thread_idx

        self.video_source_updated = False

        self.similarity_list = []
        # self.dxnn_fps = 0
        # self.sol_fps = 0

    def run(self):
        self.render_text_list()

        self.get_cap()
        while self._running:
            if self.pause_thread:
                continue

            if self.blocking_mode:
                time.sleep(1 / self.video_fps)

            self.get_cap()

            # 프레임 처리 및 시그널 전송
            rgb_image = cv2.cvtColor(self.current_original_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_qt_format.scaled(self.video_size[0], self.video_size[1], Qt.KeepAspectRatio)
            self.change_pixmap_signal.emit(self.thread_idx, p)

            # 예시 텍스트 및 FPS 업데이트
            # self.update_text_signal.emit("예시 텍스트")
            # self.update_fps_signal.emit(30.0, 60.0)

        self.cap.release()

    def stop(self):
        self._running = False
        self.wait()  # 스레드가 종료될 때까지 대기

    def get_cap(self):
        ret, frame = self.cap.read()
        if not ret:
            self.current_index = 0 if self.current_index + 1 == len(self.video_path_list) else self.current_index + 1
            self.video_path_current = os.path.join(self.base_path, self.video_path_list[self.current_index] + ".mp4")
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path_current)
            ret, frame = self.cap.read()
            self.video_source_updated = True
        self.current_original_frame = frame

    def status_video_source(self):
        ret = self.video_source_updated
        self.video_source_updated = False
        return ret

    def render_text_list(self):
        self.render_text_list_signal.emit()

    def resume(self):
        self.pause_thread = False

    def pause(self):
        self.pause_thread = True

    def update_argmax_text(self, text_list, logit_list, alarm_list):
        current_update_time_text = time.time()
        if current_update_time_text - self.last_update_time_text < self.interval_update_time_text:
            return

        argmax_info_list = []
        sorted_index = np.argsort(logit_list)
        indices_index = np.array(sorted(sorted_index[-self.number_of_alarms:]))
        for t in indices_index:
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

        self.clear_text_output_signal.emit(self.thread_idx)
        for argmax_info in argmax_info_list:
            self.update_text_output_signal.emit(self.thread_idx, argmax_info["text"], argmax_info["percent"])


        self.last_update_time_text = current_update_time_text

    def update_each_fps(self, dxnn_fps, sol_fps):
        current_update_time_fps = time.time()
        if current_update_time_fps - self.last_update_time_fps < self.interval_update_time_fps:
            return

        self.update_each_fps_signal.emit(self.thread_idx, dxnn_fps, sol_fps)

        self.last_update_time_fps = current_update_time_fps


class ClipModelSampleApp(QMainWindow):
    def __init__(self, features_path, gt_video_path_lists, gt_text_list, gt_text_alarm_level, ui_config):
        super().__init__()
        self.lock = threading.Lock()        # create lock
        self.setWindowTitle("Video Processing App")

        self.features_path = features_path
        self.gt_video_path_lists = gt_video_path_lists
        self.gt_text_list = gt_text_list
        self.gt_text_alarm_level = gt_text_alarm_level
        self.number_of_alarms = 2
        self.num_channels = len(self.gt_video_path_lists)

        self.ui_config: UIConfig = ui_config

        self.large_font = self.font()
        self.large_font.setPointSize(18)
        self.large_font_line_height = 76

        self.small_font = self.font()
        self.small_font.setPointSize(9)
        self.small_font_line_height = 38

        self.smaller_font = self.font()
        self.smaller_font.setPointSize(7)
        self.smaller_font_line_height = 30

        screen_resolution = QApplication.desktop().screenGeometry()
        self.window_w = screen_resolution.width()
        self.window_h = screen_resolution.height()

        if self.ui_config.fullscreen_mode:
            self.showFullScreen()
        else:
            self.window_h -= 100

        if self.ui_config.dark_theme:
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        print(f"Screen resolution: {self.window_w}x{self.window_h}")

        self.grid_rows, self.grid_cols = self.calculate_grid_size(self.num_channels)

        base_video_area_w = int(self.window_w * 0.7)  # 입력 터미널 UI 영역과 비디오 UI 영역의 비율 설정
        scale_factor = self.window_h / self.window_w
        calc_video_area_h = int(base_video_area_w * scale_factor)

        self.video_area_w = base_video_area_w
        self.video_area_h = calc_video_area_h

        self.resize(self.video_area_w, self.video_area_h)
        self.video_size = (int(self.video_area_w / self.grid_rows), int(self.video_area_h / self.grid_cols))

        # overall fps 를 구하기 위한 list
        self.each_fps_info_list = []

        # 위젯 초기화 및 레이아웃 설정
        self.each_fps_label_list: List[QLabel] = []
        self.video_label_list: List[QLabel] = []
        self.text_output_list: List[QTextEdit] = []

        # QLineEdit 및 QTextEdit 초기화 (텍스트 입력 및 표시)
        self.text_input_label = QLabel("[Input Terminal]")
        self.text_input = QLineEdit(self)
        self.text_input.setMinimumWidth(400)
        self.text_input.setPlaceholderText("Please enter a sentence.")
        self.text_list_label = QLabel("[Sentence List]")
        self.text_list = QTextEdit(self)
        self.text_list.setFont(self.small_font)
        self.text_list.setReadOnly(True)

        # QLabel 초기화 (FPS 표시)
        self.overall_fps_label = QLabel("FPS: N/A", self)
        self.overall_fps_label.setAlignment(Qt.AlignRight)

        # QPushButton 초기화 (비디오 시작 및 종료)
        self.start_button = QPushButton("Start", self)
        self.stop_button = QPushButton("Stop", self)

        # QPushButton 초기화 (텍스트 입력 및 삭제)
        self.add_button = QPushButton("Add", self)
        self.del_button = QPushButton("Delete", self)
        self.clear_button = QPushButton("Clear", self)

        self.init_ui()

        self.gt_text_list = gt_text_list

        # SingleVideoThread 초기화 및 시그널 연결
        div = int(np.ceil(np.sqrt(self.num_channels)))
        print("DIV : ", div)
        self.video_thread_list: List[SingleVideoThread] = []

        for thread_idx in range(self.num_channels):
            self.video_thread_list.append(
                self.set_video_thread_list(thread_idx)
            )

        # 버튼 이벤트 연결
        self.start_button.clicked.connect(self.resume)
        self.stop_button.clicked.connect(self.pause)
        self.add_button.clicked.connect(self.push_text_list)
        self.del_button.clicked.connect(self.pop_text_list)
        self.clear_button.clicked.connect(self.clear_text_list)

        # text_input 이벤트 연결
        self.text_input.returnPressed.connect(self.push_text_list)  # enter key

        # text embedding vector 미리 준비
        with self.lock:
            self.text_vector_list = TextVectorUtil.get_text_vector_list(gt_text_list)

        self.dxnn_video_encoder = DXVideoEncoder(get_args().video_encoder_dxnn)

        # DXEngineThread 초기화
        self.dx_engine_thread = DXEngineThread(self)
        self.dx_engine_thread.update_overall_fps_signal.connect(self.update_overall_fps)

        self.start()

    def init_ui(self):
        # 레이아웃 설정
        video_layout = QVBoxLayout()
        video_box = self.generate_video_box()
        video_layout.addLayout(video_box)

        control_layout = QVBoxLayout()
        button_box = self.generate_control_ui()
        control_layout.addLayout(button_box)

        if self.ui_config.terminal_mode:
            terminal_box = self.generate_terminal_box()
            control_layout.addLayout(terminal_box)

            app_layout = QHBoxLayout()
            app_layout.addLayout(video_layout)
            app_layout.addLayout(control_layout)
        else:
            app_layout = QVBoxLayout()
            app_layout.addLayout(control_layout)
            app_layout.addLayout(video_layout)

        container = QWidget()
        container.setLayout(app_layout)
        self.setCentralWidget(container)

    def generate_terminal_box(self):
        # terminal layout
        # [text label]
        # [text input field] | [add] [del] [clr]
        # --------------------------------
        # [
        # text list
        # ...
        # ]
        # --------------------------------
        terminal_box = QVBoxLayout()
        terminal_box.addWidget(self.text_input_label)
        input_box = QHBoxLayout()
        input_box.addWidget(self.text_input)
        input_box.addWidget(self.add_button)
        input_box.addWidget(self.del_button)
        input_box.addWidget(self.clear_button)
        terminal_box.addLayout(input_box)
        terminal_box.addWidget(self.text_list_label)
        terminal_box.addWidget(self.text_list)
        return terminal_box

    def generate_control_ui(self):
        # start / stop 버튼 & fps display 표시
        # [start] [stop] | [fps info]
        control_box = QHBoxLayout()
        control_box.addWidget(self.start_button)
        control_box.addWidget(self.stop_button)
        control_box.addWidget(self.overall_fps_label)
        if self.ui_config.fullscreen_mode:
            exit_button = QPushButton("Exit", self)
            exit_button.clicked.connect(self.close_application)
            control_box.addWidget(exit_button)
        return control_box

    @staticmethod
    def calculate_grid_size(num_channels):
        # 가로와 세로 비율을 고려한 그리드 크기 결정
        if num_channels == 1:
            return 1, 1
        elif num_channels == 2:
            return 2, 1
        elif num_channels == 3:
            return 3, 1
        elif num_channels == 4:
            return 2, 2
        elif num_channels <= 6:
            return 3, 2
        elif num_channels <= 8:
            return 4, 2
        elif num_channels <= 9:
            return 3, 3
        elif num_channels <= 12:
            return 4, 3
        elif num_channels <= 16:
            return 4, 4
        else:
            return math.ceil(num_channels ** 0.5) if num_channels > 1 else 1  # 1일 경우 그리드 필요 없음

    def generate_video_box(self):

        if self.num_channels == 1:
            video_layout = QVBoxLayout()

            if self.ui_config.show_each_fps_label:
                # 1 채널일 경우 단일 영상과 텍스트 출력
                each_fps_label = QLabel("", self)
                each_fps_label.setFont(self.small_font)
                each_fps_label.setFixedHeight(self.small_font_line_height)
                each_fps_label.setAlignment(Qt.AlignRight)
                self.each_fps_label_list.append(each_fps_label)
                video_layout.addWidget(each_fps_label)
            self.each_fps_info_list.append({"dxnn_fps": -1, "sol_fps": -1})

            video_label = QLabel(self)
            video_label.setAlignment(Qt.AlignCenter)
            self.video_label_list.append(video_label)
            video_layout.addWidget(video_label)

            text_output = QTextEdit(self)
            text_output.setFont(self.large_font)
            text_output.setFixedHeight(self.large_font_line_height * self.number_of_alarms)
            text_output.setAlignment(Qt.AlignCenter)
            text_output.setReadOnly(True)
            self.text_output_list.append(text_output)
            video_layout.addWidget(text_output)

            return video_layout
        else:
            # 멀티 채널일 경우 grid 생성
            video_grid_layout = QGridLayout()

            for i in range(self.num_channels):

                video_layout = QVBoxLayout()

                if self.ui_config.show_each_fps_label:
                    each_fps_label = QLabel("FPS: N/A", self)
                    each_fps_label.setFont(self.smaller_font)
                    each_fps_label.setFixedHeight(self.smaller_font_line_height)
                    each_fps_label.setAlignment(Qt.AlignRight)
                    self.each_fps_label_list.append(each_fps_label)
                    video_layout.addWidget(each_fps_label)
                self.each_fps_info_list.append({"dxnn_fps": -1, "sol_fps": -1})     # for calculate overall fps

                video_label = QLabel(self)
                video_label.setAlignment(Qt.AlignCenter)
                video_layout.addWidget(video_label)

                text_output = QTextEdit(self)
                text_output.setFont(self.small_font)
                text_output.setFixedWidth(self.video_size[0])
                text_output.setFixedHeight(self.small_font_line_height * self.number_of_alarms)
                # text_output.setStyleSheet("background-color: black; color: white;")
                text_output.setAlignment(Qt.AlignCenter)
                text_output.setReadOnly(True)
                video_layout.addWidget(text_output)

                video_grid_layout.addLayout(video_layout, i // self.grid_cols, i % self.grid_cols)

                self.text_output_list.append(text_output)
                self.video_label_list.append(video_label)

            return video_grid_layout

    def set_video_thread_list(self, thread_idx):
        single_video_thread = SingleVideoThread(self.features_path, self.gt_video_path_lists[thread_idx], thread_idx,
                                                self.video_size, self.number_of_alarms)
        # signal 바인딩
        single_video_thread.change_pixmap_signal.connect(self.update_video)
        single_video_thread.clear_text_output_signal.connect(self.clear_text_output)
        single_video_thread.update_text_output_signal.connect(self.update_text_output)
        single_video_thread.render_text_list_signal.connect(self.render_text_list)
        single_video_thread.update_each_fps_signal.connect(self.update_each_fps)
        return single_video_thread

    def start(self):
        for video_thread in self.video_thread_list:
            video_thread.start()
            video_thread.similarity_list = np.zeros((len(self.gt_text_list)))
        self.dx_engine_thread.start()

    def resume(self):
        for video_thread in self.video_thread_list:
            video_thread.resume()
        self.dx_engine_thread.resume()

    def pause(self):
        for video_thread in self.video_thread_list:
            video_thread.pause()
        self.dx_engine_thread.pause()

    def update_video(self, idx, qt_img):
        self.video_label_list[idx].setPixmap(QPixmap.fromImage(qt_img))

    def clear_text_output(self, idx: int):
        self.text_output_list[idx].clear()

    def update_text_output(self, idx: int, text: str, progress: int):
        if self.ui_config.show_percent:
            self.text_output_list[idx].append(" [" + str(progress) + "%] " + text)
        else:
            self.text_output_list[idx].append(text)

    def show_toast(self, text, title="Info", duration=3000, preset=ToastPreset.WARNING, position=ToastPosition.CENTER):
        toast = Toast(self)
        toast.setDuration(duration)
        toast.setTitle(title)
        toast.setText(text)
        toast.applyPreset(preset)
        Toast.setPosition(position)
        toast.show()

    def render_text_list(self):
        self.text_list.clear()
        for gt_text_item in self.gt_text_list:
            self.text_list.append(gt_text_item)

    def push_text_list(self):
        if len(self.text_input.text()) == 0:
            self.show_toast("Please enter a sentence and press the Add button.")
            return

        with self.lock:
            self.gt_text_list.append(self.text_input.text())
            self.text_vector_list = TextVectorUtil.get_text_vector_list(self.gt_text_list)
            self.gt_text_alarm_level.append([0.23, 0.31, 0.26])
            self.init_similarity_list()

            self.text_input.clear()
            self.render_text_list()

    def pop_text_list(self):
        with self.lock:
            if len(self.gt_text_list) > 0:
                self.gt_text_list.pop(-1)
                self.text_vector_list.pop(-1)
                self.gt_text_alarm_level.pop(-1)
                self.init_similarity_list()

                self.render_text_list()
            else:
                self.show_toast("No sentences to delete.")

    def init_similarity_list(self):
        for video_thread in self.video_thread_list:
            video_thread.similarity_list = np.zeros((len(self.gt_text_list)))

    def clear_text_list(self):
        with self.lock:
            self.gt_text_list.clear()
            self.text_vector_list.clear()
            self.gt_text_alarm_level.clear()
            self.init_similarity_list()

            self.render_text_list()

    def update_overall_fps(self):
        if len(self.each_fps_info_list) > 0:
            sum_overall_dxnn_fps = 0
            sum_overall_sol_fps = 0
            for each_fps in self.each_fps_info_list:
                sum_overall_dxnn_fps += each_fps["dxnn_fps"]
                sum_overall_sol_fps += each_fps["sol_fps"]

            overall_dxnn_fps = sum_overall_dxnn_fps / len(self.each_fps_info_list)
            overall_sol_fps = sum_overall_sol_fps / len(self.each_fps_info_list)

            self.overall_fps_label.setText(
                f" NPU FPS: {overall_dxnn_fps:.2f}, Render FPS: {overall_sol_fps:.2f} ")

    def update_each_fps(self, thread_idx, dxnn_fps, sol_fps):
        if self.ui_config.show_each_fps_label:
            self.each_fps_label_list[thread_idx].setText(f" NPU FPS: {dxnn_fps:.2f}, Render FPS: {sol_fps:.2f} ")

        self.each_fps_info_list[thread_idx]["dxnn_fps"] = dxnn_fps
        self.each_fps_info_list[thread_idx]["sol_fps"] = sol_fps

    @overrides
    def closeEvent(self, event):
        # 모든 비디오 스레드를 종료
        for video_thread in self.video_thread_list:
            video_thread.stop()

        self.dx_engine_thread.stop()
        super().closeEvent(event)

    def close_application(self):
        self.close()

class DXEngineThread(QThread):
    update_overall_fps_signal = pyqtSignal()

    def __init__(self, ctx: ClipModelSampleApp):
        super().__init__()
        self._running = True
        self.ctx: ClipModelSampleApp = ctx
        self.image_transform = self.transform(224)
        self.video_mask = torch.ones(1, 1)
        self.pause_thread = False

    @staticmethod
    def transform(n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @overrides()
    def run(self):
        time.sleep(0.1)
        frame_count = 0

        while self._running:
            if self.pause_thread:
                continue

            for index in range(len(self.ctx.video_thread_list)):
                s = time.perf_counter_ns()

                similarity_list = []
                vCap = self.ctx.video_thread_list[index]
                if vCap.status_video_source():
                    vCap.similarity_list = np.zeros((len(self.ctx.gt_text_list)))
                    vCap.last_update_time_text = 0
                    frame_count = 0
                frame = vCap.current_original_frame.copy()
                dxnn_s = time.perf_counter_ns()
                input_data = self.image_transform(Image.fromarray(frame).convert("RGB"))
                video_pred = self.ctx.dxnn_video_encoder.run(input_data)[0]
                dxnn_e = time.perf_counter_ns()
                # print(index, " : ", video_pred.shape)

                with self.ctx.lock:
                    for text_index in range(len(self.ctx.text_vector_list)):
                        ret = self._loose_similarity(self.ctx.text_vector_list[text_index], video_pred, self.video_mask)
                        similarity_list.append(ret)
                    try:
                        if len(similarity_list) > 0:
                            similarity_list = np.stack(similarity_list).reshape(len(self.ctx.text_vector_list))
                    except Exception as ex:
                        print(ex)
                    vCap.similarity_list += similarity_list

                e = time.perf_counter_ns()
                dxnn_fps = 1000 / ((dxnn_e - dxnn_s) / 1000000)
                sol_fps = 1000 / ((e - s) / 1000000)
                vCap.dxnn_fps = dxnn_fps
                vCap.sol_fps = sol_fps

                vCap.update_each_fps(dxnn_fps, sol_fps)

            for index in range(len(self.ctx.video_thread_list)):
                with self.ctx.lock:
                    vCap = self.ctx.video_thread_list[index]
                    vCap.update_argmax_text(self.ctx.gt_text_list, vCap.similarity_list / (frame_count + 1),
                                            self.ctx.gt_text_alarm_level)

            self.update_overall_fps()
            frame_count += 1

    def stop(self):
        self._running = False
        self.wait()  # 스레드가 종료될 때까지 대기

    def resume(self):
        self.pause_thread = False

    def pause(self):
        self.pause_thread = True

    def update_overall_fps(self):
        self.update_overall_fps_signal.emit()

    @staticmethod
    def _mean_pooling_for_similarity_visual(vis_output, video_frame_mask):
        video_mask_un = video_frame_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = vis_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.0] = 1.0
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _loose_similarity(self, text_vectors, video_vectors, video_frame_mask):
        sequence_output, visual_output = (
            text_vectors.contiguous(),
            video_vectors.contiguous(),
        )
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(
            visual_output, video_frame_mask
        )
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        try:
            if sequence_output.ndim > 1:
                sequence_output = sequence_output.squeeze(1)
        except Exception as ex:
            print(ex)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        retrieve_logits = torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif sys.platform.startswith("darwin"):
        return "mps"
    else:
        return "cpu"

class TextVectorUtil:
    SPECIAL_TOKEN = {
        "CLS_TOKEN": "<|startoftext|>",
        "SEP_TOKEN": "<|endoftext|>",
        "MASK_TOKEN": "[MASK]",
        "UNK_TOKEN": "[UNK]",
        "PAD_TOKEN": "[PAD]",
    }
    MAX_WORDS = 32
    DEVICE = get_device()

    model_load_time_s = time.perf_counter_ns()

    token_embedder = PiaONNXTensorRTModel(
        model_path=get_args().token_embedder_onnx, device=DEVICE
    )
    text_encoder = PiaONNXTensorRTModel(
        model_path=get_args().text_encoder_onnx, device=DEVICE
    )

    model_load_time_e = time.perf_counter_ns()
    print("[TIME] Model Load : {} ns".format(model_load_time_e - model_load_time_s))



    @staticmethod
    def get_text_vector_list(text_list: List[str]):
        ret = []
        for i in tqdm(range(len(text_list))):
            text = text_list[i]
            token = ClipTokenizer().tokenize(text)
            token = [TextVectorUtil.SPECIAL_TOKEN["CLS_TOKEN"]] + token
            total_length_with_class = TextVectorUtil.MAX_WORDS - 1
            if len(token) > total_length_with_class:
                token = token[:total_length_with_class]
            token = token + [TextVectorUtil.SPECIAL_TOKEN["SEP_TOKEN"]]
            token_ids = ClipTokenizer().convert_tokens_to_ids(token)
            token_ids_mask = [1] * len(token_ids) + [0] * (
                    TextVectorUtil.MAX_WORDS - len(token_ids)
            )
            token_ids = token_ids + [0] * (
                    TextVectorUtil.MAX_WORDS - len(token_ids)
            )
            token_ids_mask = torch.tensor([token_ids_mask]).to(TextVectorUtil.DEVICE, dtype=torch.float32)
            token_ids = torch.tensor([token_ids]).to(TextVectorUtil.DEVICE)
            text_embedding = TextVectorUtil.token_embedder(token_ids)
            text_vectors = TextVectorUtil.text_encoder([text_embedding, token_ids_mask])
            ret.append(text_vectors)
        return ret if len(ret) > 1 else text_vectors


class UIConfig:
    show_percent = False
    show_each_fps_label = False
    terminal_mode = True
    fullscreen_mode = True
    dark_theme = True


class SettingsWindow(QMainWindow):
    gt_video_path_lists = [
        [
            "demo_videos/fire_on_car",
        ],
        [
            "demo_videos/dam_explosion_short",
        ],
        [
            "demo_videos/violence_in_shopping_mall_short",
        ],
        [
            "demo_videos/gun_terrorism_in_airport",
        ],
        [
            "demo_videos/crowded_in_subway",
        ],
        [
            "demo_videos/heavy_structure_falling",
        ],
        [
            "demo_videos/electrical_outlet_is_emitting_smoke",
        ],
        [
            "demo_videos/pot_is_catching_fire",
        ],
        [
            "demo_videos/falldown_on_the_grass",
        ],
        [
            "demo_videos/fighting_on_field",
        ],
        [
            "demo_videos/fire_in_the_kitchen",
        ],
        [
            "demo_videos/group_fight_on_the_streat",
        ],
        [
            "demo_videos/iron_is_on_fire",
        ],
        [
            "demo_videos/someone_helps_old_man_who_is_fallting_down",
        ],
        [
            "demo_videos/the_pile_of_sockets_is_smoky_and_on_fire"
        ],
        [
            "demo_videos/two_childrens_are_fighting",
        ],
    ]
    gt_text_alarm_level = [
        [0.27, 0.29, 0.28],  # "The subway is crowded with people",
        [0.27, 0.29, 0.28],  # "People is crowded in the subway",

        [0.21, 0.25, 0.225],  # "Heavy objects are fallen",

        [0.23, 0.25, 0.24],  # "Physical confrontation occurs between two people",
        [0.22, 0.25, 0.23],  # "Violence with kicking and punching",

        [0.27, 0.29, 0.28],  # "Terrorism is taking place at the airport",
        [0.23, 0.26, 0.247],  # "Terrorist is shooting at people",

        [0.24, 0.28, 0.255],  # "The water is exploding out",
        [0.24, 0.28, 0.255],  # "The water is gushing out",

        [0.23, 0.26, 0.24],  # "Fire is coming out of the car",
        [0.24, 0.28, 0.26],  # "The car is exploding",

        [0.23, 0.26, 0.24],  # "The electrical outlet on the wall is emitting smoke",
        [0.23, 0.26, 0.24],  # "Smoke is rising from the electrical outlet."

        [0.23, 0.26, 0.24],  # "A pot on the induction cooktop is catching fire.",
        [0.23, 0.26, 0.24],  # "A fire broke out in a pot in the kitchen."

        [0.23, 0.26, 0.24],  # "Two childrens are fighting.",
        [0.23, 0.26, 0.24],  # "Two children start crying after a fight."

        [0.23, 0.26, 0.24],  # "Several men are engaged in a fight.",
        [0.23, 0.26, 0.24],  # "Several people are fighting in the street.",

        [0.23, 0.26, 0.24],  # "An elderly man is complaining of pain on the street."
        [0.23, 0.26, 0.24],  # "An man is crouching on the street."

        [0.23, 0.26, 0.24],  # "Someone helps old man who is falling down."
        [0.23, 0.26, 0.24],  # "An elderly grandfather is lying on the floor."

        [0.23, 0.26, 0.24],  # "A fire has occurred in the electric iron."
        [0.23, 0.26, 0.24],  # "The electric iron on the table is on fire."

        [0.23, 0.26, 0.24],  # "Two men are engaging in mixed martial arts on the ring."
    ]

    gt_text_list = [
        "The subway is crowded with people",
        "People is crowded in the subway",

        "Heavy objects are fallen",

        "Physical confrontation occurs between two people",
        "Violence with kicking and punching",

        "Terrorism is taking place at the airport",
        "Terrorist is shooting at people",

        "The water is exploding out",
        "The water is gushing out",

        "Fire is coming out of the car",
        "The car is exploding",

        "The electrical outlet on the wall is emitting smoke",
        "Smoke is rising from the electrical outlet.",

        "A pot on the induction cooktop is catching fire.",
        "A fire broke out in a pot in the kitchen.",

        "Two childrens are fighting.",
        "Two children start crying after a fight.",

        "Several men are engaged in a fight.",
        "Several people are fighting in the street.",

        "An elderly man is complaining of pain on the street.",
        "An man is crouching on the street.",

        "Someone helps old man who is falling down.",
        "An elderly grandfather is lying on the floor",

        "A fire has occurred in the electric iron.",
        "The electric iron on the table is on fire.",

        "Two men are engaging in mixed martial arts on the ring.",
    ]

    def __init__(self, args):
        super().__init__()

        # CLI arguments
        self.features_path = args.features_path
        self.number_of_channels = args.number_of_channels
        self.max_words = args.max_words
        self.feature_framerate = args.feature_framerate
        self.slice_framepos = args.slice_framepos

        # input data setting
        self.gt_video_path_lists = self.gt_video_path_lists
        self.adjusted_video_path_lists = None
        self.gt_text_list = self.gt_text_list
        self.gt_text_alarm_level = self.gt_text_alarm_level

        # UI config
        self.ui_config: UIConfig = UIConfig()

        self.setWindowTitle("Settings")
        self.setMinimumWidth(1000)

        # 레이아웃 설정
        layout = QVBoxLayout()

        # features_path 입력
        self.features_path_label = QLabel("Features Path:")
        self.features_path_input = QLineEdit(self)
        self.features_path_input.setText(self.features_path)
        layout.addWidget(self.features_path_label)
        layout.addWidget(self.features_path_input)

        # number_of_channels 설정
        self.number_of_channels_label = QLabel("Number of Channels:")
        self.number_of_channels_input = QSpinBox(self)
        self.number_of_channels_input.setValue(self.number_of_channels)
        self.number_of_channels_input.setRange(1, 16)  # 1 ~ 16 채널로 제한
        layout.addWidget(self.number_of_channels_label)
        layout.addWidget(self.number_of_channels_input)

        # Show Percentage
        self.show_percent_checkbox = QCheckBox("Display Percentage", self)
        self.show_percent_checkbox.setChecked(self.ui_config.show_percent)
        layout.addWidget(self.show_percent_checkbox)

        # Show FPS Label
        self.show_each_fps_label_checkbox = QCheckBox("Display FPS for each video", self)
        self.show_each_fps_label_checkbox.setChecked(self.ui_config.show_each_fps_label)
        layout.addWidget(self.show_each_fps_label_checkbox)

        # Terminal Mode
        self.terminal_mode_checkbox = QCheckBox("Terminal Mode", self)
        self.terminal_mode_checkbox.setChecked(self.ui_config.terminal_mode)
        layout.addWidget(self.terminal_mode_checkbox)

        # Fullscreen Mode
        self.fullscreen_mode_checkbox = QCheckBox("Fullscreen Mode", self)
        self.fullscreen_mode_checkbox.setChecked(self.ui_config.fullscreen_mode)
        layout.addWidget(self.fullscreen_mode_checkbox)

        # Dark Theme
        self.dark_theme_checkbox = QCheckBox("Dark Theme", self)
        self.dark_theme_checkbox.setChecked(self.ui_config.dark_theme)
        layout.addWidget(self.dark_theme_checkbox)

        # Done 버튼
        self.done_button = QPushButton("Done", self)
        self.done_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.done_button)

        # 창에 레이아웃 설정
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def apply_settings(self):
        # 사용자가 입력한 값 적용
        self.features_path = self.features_path_input.text()
        self.adjusted_video_path_lists = self.adjust_video_path_lists(self.gt_video_path_lists, self.number_of_channels_input.value())
        self.ui_config.show_percent = self.show_percent_checkbox.isChecked()
        self.ui_config.show_each_fps_label = self.show_each_fps_label_checkbox.isChecked()
        self.ui_config.terminal_mode = self.terminal_mode_checkbox.isChecked()
        self.ui_config.fullscreen_mode = self.fullscreen_mode_checkbox.isChecked()
        self.ui_config.dark_theme = self.dark_theme_checkbox.isChecked()

        # 설정 창을 닫고 메인 앱 실행
        self.close()
        self.start_main_app()

    def start_main_app(self):
        # 메인 애플리케이션 실행
        self.main_app = ClipModelSampleApp(self.features_path, self.adjusted_video_path_lists, self.gt_text_list, self.gt_text_alarm_level, self.ui_config)
        self.main_app.show()

    def adjust_video_path_lists(self, gt_video_path_lists, number_of_channels):
        if number_of_channels <= 1:
            gt_video_path_lists_for_single = []
            enriched_path_list = []
            for path_list in gt_video_path_lists:
                for path in path_list:
                    enriched_path_list.append(path)
            gt_video_path_lists_for_single.append(enriched_path_list)
            return gt_video_path_lists_for_single
        else:
            return gt_video_path_lists[:number_of_channels]

def main():
    # Get Input Arguments
    args = get_args()

    app = QApplication(sys.argv)

    # 설정 창 실행
    settings_window = SettingsWindow(args)
    settings_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
