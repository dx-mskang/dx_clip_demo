import logging
import os
import threading
import time
from typing import List, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage


class VideoProducer(QObject):
    __scaled_video_frame_updated_signal = pyqtSignal(int, QImage)
    __origin_video_frame_updated_signal = pyqtSignal(int, np.ndarray, int)
    __video_source_changed_signal = pyqtSignal(int)

    def __init__(self, channel_idx: int, base_path: str, video_path_list: List[str], video_size: Tuple,
                 video_fps_sync_mode: int, video_frame_skip_interval: int, sentence_list_updated_signal: pyqtSignal):
        super().__init__()
        self.__running = True
        self.__pause_thread = False
        self.__video_fps_sync_mode = video_fps_sync_mode

        self.__change_video_lock = threading.Lock()

        self.__frame_count = 0
        self.__video_frame_skip_interval = video_frame_skip_interval

        sentence_list_updated_signal.connect(self.__change_video)

        self._channel_idx = channel_idx

        self.__current_index = 0
        if video_path_list[0] == "/dev/video0":
            self.__base_path = ""
            self.video_path_list = ["/dev/video0"]
            self.__video_path_current = os.path.join(self.video_path_list[self.__current_index])
            self.__is_camera_source = True
        else:
            self.__base_path = base_path
            self.video_path_list = video_path_list
            self.__video_path_current = os.path.join(self.__base_path, self.video_path_list[self.__current_index] + ".mp4")
            self.__is_camera_source = False

        self.__video_size = video_size
        self.__video_label_size = video_size

        self.__video_capture = cv2.VideoCapture(self.__video_path_current)

        if not self.__video_capture:
            logging.error("Error: Could not open video.")
            self.__video_fps = 30       # default video fps value
        else:
            self.__video_fps = int(round(self.__video_capture.get(cv2.CAP_PROP_FPS), 0))
            logging.debug("channel_idx:" + str(self._channel_idx) + f"FPS: {self.__video_fps}")

        self.__current_video_frame = np.zeros((self.__video_label_size[1], self.__video_label_size[0], 3), dtype=np.uint8)

    def is_camera_source(self):
        return self.__is_camera_source

    def set_video_label_size(self, size: Tuple[int, int]):
        self.__video_label_size = size

    def capture_frame(self):
        logging.debug("VideoProducer thread started, channel_id: " + str(self._channel_idx))

        if self.__running is False or self.__pause_thread:
            time.sleep(0.1)  # Adding a small sleep to avoid busy-waiting
            return None

        if self.__video_fps_sync_mode:
            # 1: 1sec, 0.75: Adjustment ratio considering NPU inference execution time
            time.sleep((1 / self.__video_fps) * 0.75)

        self.__frame_count += 1
        if self.__video_frame_skip_interval > 0:
            if self.__frame_count % self.__video_frame_skip_interval == 0:
                self.__frame_count = 0
                return None

        self.__update_current_video_frame()

        # Frame processing and signal transmission
        rgb_image = cv2.cvtColor(self.__current_video_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_image = convert_to_qt_format.scaled(self.__video_label_size[0], self.__video_label_size[1], Qt.KeepAspectRatio)

        # Send the scaled QImage to the main thread
        self.__scaled_video_frame_updated_signal.emit(self._channel_idx, scaled_image)

        # Send the original QImage to the video consumer thread
        return [self._channel_idx, self.__current_video_frame, self.__video_fps]

    def __update_current_video_frame(self):
        with self.__change_video_lock:
            ret, frame = self.__video_capture.read()
        if not ret:
            self.__change_video(True)
        else:
            self.__current_video_frame = frame

    def __change_video(self, is_next=False):
        camera_mode = False
        if self.video_path_list[0] == '/dev/video0':
            camera_mode = True
            is_next = False

        if is_next:
            self.__current_index = 0 if self.__current_index + 1 == len(self.video_path_list) else self.__current_index + 1

        if camera_mode:
            self.__video_path_current = os.path.join(self.video_path_list[self.__current_index])
        else:
            self.__video_path_current = os.path.join(self.__base_path,
                                                     self.video_path_list[self.__current_index] + ".mp4")
        with self.__change_video_lock:
            self.__video_capture.release()
            self.__video_capture = cv2.VideoCapture(self.__video_path_current)
            ret, frame = self.__video_capture.read()
            if ret:
                self.__video_source_changed_signal.emit(self._channel_idx)
                self.__current_video_frame = frame
            else:
                logging.debug("fail to read video frame : " + str(ret))

    def stop(self):
        self.__running = False
        self.__pause_thread = False  # Ensure thread is not stuck in pause
        logging.debug(f"VideoProducer {self._channel_idx} stopping...")
        with self.__change_video_lock:
            if self.__video_capture.isOpened():
                self.__video_capture.release()

    def resume(self):
        self.__pause_thread = False

    def pause(self):
        self.__pause_thread = True

    def get_video_frame_updated_signal(self) -> [pyqtSignal]:
        return [self.__scaled_video_frame_updated_signal, self.__video_source_changed_signal]
