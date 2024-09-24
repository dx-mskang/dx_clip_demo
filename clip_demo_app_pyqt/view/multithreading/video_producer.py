import copy
import os
import threading
import time
from typing import List, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage


class VideoProducer(QThread):
    __scaled_video_frame_updated_signal = pyqtSignal(int, QImage)
    __origin_video_frame_updated_signal = pyqtSignal(int, np.ndarray)
    __video_source_changed_signal = pyqtSignal(int)

    def __init__(self, channel_idx: int, base_path: str, video_path_list: List[str], video_size: Tuple,
                 sentence_list_updated_signal: pyqtSignal):
        super().__init__()
        self.__running = True
        self.__pause_thread = False
        self.__blocking_mode = True
        # self.__blocking_mode = False

        self.__next_video_lock = threading.Lock()

        sentence_list_updated_signal.connect(self.__next_video)

        self.__channel_idx = channel_idx

        self.__current_index = 0
        if video_path_list[0] == "/dev/video0":
            self.__base_path = ""
            self.__video_path_list = ["/dev/video0"]
            self.__video_path_current = os.path.join(self.__video_path_list[self.__current_index])
        else:
            self.__base_path = base_path
            self.__video_path_list = video_path_list
            self.__video_path_current = os.path.join(self.__base_path, self.__video_path_list[self.__current_index] + ".mp4")

        self.__video_size = video_size
        self.__video_fps = 30

        self.__video_capture = cv2.VideoCapture(self.__video_path_current)
        self.__current_video_frame = np.zeros((self.__video_size[1], self.__video_size[0], 3), dtype=np.uint8)

    def run(self):
        # print("run", QThread.currentThread())

        self.__update_current_video_frame()
        while self.__running:
            if self.__pause_thread:
                continue

            if self.__blocking_mode:
                time.sleep(1 / self.__video_fps)

            self.__update_current_video_frame()

            # 프레임 처리 및 시그널 전송
            rgb_image = cv2.cvtColor(self.__current_video_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_image = convert_to_qt_format.scaled(self.__video_size[0], self.__video_size[1], Qt.KeepAspectRatio)

            # Scaled QImage를 메인 스레드로 전송
            self.__scaled_video_frame_updated_signal.emit(self.__channel_idx, scaled_image)

            # Original QImage를 video Consumer 스레드로 전송
            self.__origin_video_frame_updated_signal.emit(self.__channel_idx, copy.deepcopy(self.__current_video_frame))

        with self.__next_video_lock:
            self.__video_capture.release()

    def __update_current_video_frame(self):
        with self.__next_video_lock:
            ret, frame = self.__video_capture.read()
        if not ret:
            self.__next_video()
        else:
            self.__current_video_frame = frame

    def __next_video(self):

        self.__current_index = 0 if self.__current_index + 1 == len(self.__video_path_list) else self.__current_index + 1
        self.__video_path_current = os.path.join(self.__base_path,
                                                 self.__video_path_list[self.__current_index] + ".mp4")
        with self.__next_video_lock:
            self.__video_capture.release()
            self.__video_capture = cv2.VideoCapture(self.__video_path_current)
            ret, frame = self.__video_capture.read()
            if ret:
                self.__video_source_changed_signal.emit(self.__channel_idx)
                self.__current_video_frame = frame
            else:
                print("fail to read video frame : " + str(ret))

    def get_current_video_frame(self):
        return self.__current_video_frame

    def stop(self):
        self.__running = False
        self.wait()

    def resume(self):
        self.__pause_thread = False

    def pause(self):
        self.__pause_thread = True

    def get_video_frame_updated_signal(self) -> [pyqtSignal]:
        return [self.__scaled_video_frame_updated_signal, self.__origin_video_frame_updated_signal,
                self.__video_source_changed_signal]
