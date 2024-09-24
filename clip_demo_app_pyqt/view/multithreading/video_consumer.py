from abc import abstractmethod
import time
from queue import Queue

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from clip_demo_app_pyqt.common.config.ui_config import UIConfig


class VideoConsumer(QThread):
    __update_each_fps_signal = pyqtSignal(int, float, float)
    __update_overall_fps_signal = pyqtSignal()

    def __init__(self, channel_idx: int, origin_video_frame_updated_signal: pyqtSignal,
                 video_source_changed_signal: pyqtSignal):
        super().__init__()
        self.__running = True
        self.__pause_thread = False
        self.__max_queue_size = UIConfig.video_consumer_queue_size
        self.__queue: Queue[np.ndarray] = Queue(maxsize=self.__max_queue_size)

        self._channel_idx = channel_idx

        self.__last_update_time_each_fps = 0  # Initialize the last update time
        self.__interval_update_time_each_fps = 0.3  # Set update interval to 0.3 seconds (adjust as needed)

        self.__last_update_time_overall_fps = 0  # Initialize the last update time
        self.__interval_update_time_overall_fps = 0.3  # Set update interval to 0.3 seconds (adjust as needed)

        origin_video_frame_updated_signal.connect(self.__push_origin_video_frame)
        video_source_changed_signal.connect(self.__video_source_changed)

    def run(self):
        # TODO : 불필요 여부 확인 필요
        # self.render_text_list()
        while self.__running:
            if self.__pause_thread:
                time.sleep(0.01)  # Introduce a short sleep to prevent tight looping
                continue

            frame = self.__pop_origin_video_frame()
            if frame is None:
                time.sleep(0.01)  # Introduce a short sleep to prevent tight looping
                continue
            else:
                self.process(frame)

    def stop(self):
        self.__running = False
        self.wait()

    # TODO : 불필요 여부 확인 필요
    # def render_text_list(self):
    #     self.render_text_list_signal.emit()

    def resume(self):
        self.__pause_thread = False

    def pause(self):
        self.__pause_thread = True

    def get_update_each_fps_signal(self):
        return self.__update_each_fps_signal

    def get_update_overall_fps_signal(self):
        return self.__update_overall_fps_signal

    def _update_overall_fps(self):
        current_update_time_fps = time.time()
        if current_update_time_fps - self.__last_update_time_overall_fps < self.__interval_update_time_overall_fps:
            return

        self.__update_overall_fps_signal.emit()

        self.__last_update_time_overall_fps = current_update_time_fps

    def _update_each_fps(self, dxnn_fps, sol_fps):
        current_update_time_fps = time.time()
        if current_update_time_fps - self.__last_update_time_each_fps < self.__interval_update_time_each_fps:
            return

        self.__update_each_fps_signal.emit(self._channel_idx, dxnn_fps, sol_fps)

        self.__last_update_time_each_fps = current_update_time_fps

    def __push_origin_video_frame(self, channel_idx: int, origin_video_frame: np.ndarray):
        if self._channel_idx != channel_idx:
            raise RuntimeError("channel index is not correct")

        # If the queue is full, remove the oldest image
        if self.__queue.full():
            self.__queue.get()  # Remove the oldest image

        # Add the new image to the queue
        self.__queue.put(origin_video_frame)

    def __video_source_changed(self, channel_idx: int):
        if self._channel_idx != channel_idx:
            raise RuntimeError("channel index is not correct")

        self._cleanup()

    def __pop_origin_video_frame(self):
        if not self.__queue.empty():
            return self.__queue.get()
        return None

    @abstractmethod
    def process(self, frame):
        pass

    @abstractmethod
    def _cleanup(self):
        pass

