import threading
from typing import List

import qdarkstyle
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMainWindow, QTextEdit, QLineEdit, \
    QHBoxLayout, QGridLayout
from overrides import overrides
from pyqttoast import ToastPreset, ToastPosition, Toast

from clip_demo_app_pyqt.common.base import CombinedMeta, Base
from clip_demo_app_pyqt.viewmodel.clip_view_model import ClipViewModel
from clip_demo_app_pyqt.common.config.ui_config import UIHelper, UIConfig
from clip_demo_app_pyqt.view.multithreading.clip_video_consumer import ClipVideoConsumer
from clip_demo_app_pyqt.view.multithreading.video_producer import VideoProducer
from clip_demo_app_pyqt.view.multithreading.video_worker import VideoWorker


class ClipView(Base, QMainWindow, metaclass=CombinedMeta):
    def __init__(self, view_model: ClipViewModel, ui_config: UIConfig, base_path, adjusted_video_path_lists):
        QMainWindow.__init__(self)
        QObject.__init__(self)
        self.__fps_lock = threading.Lock()
        self.__view_model = view_model

        self.base_path = base_path
        self.adjusted_video_path_lists = adjusted_video_path_lists
        self.ui_config = ui_config

        self.ui_helper = UIHelper(self, self.ui_config)

        self.setWindowTitle("Video Processing App")

        if self.ui_helper.ui_config.dark_theme:
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        if self.ui_helper.ui_config.fullscreen_mode:
            self.showFullScreen()

        self.resize(self.ui_helper.video_area_w, self.ui_helper.video_area_h)

        # List for calculating overall FPS
        self.each_fps_info_list = []

        # Widget initialization
        self.each_fps_label_list: List[QLabel] = []
        self.video_label_list: List[QLabel] = []
        self.text_output_list: List[QTextEdit] = []

        # QLineEdit and QTextEdit initialization (text input and display)
        self.sentence_input_label = QLabel("[Input Terminal]")
        self.sentence_input = QLineEdit(self)
        self.sentence_input.setMinimumWidth(self.ui_config.sentence_input_min_width)
        self.sentence_input.setPlaceholderText("Please enter a sentence.")
        self.sentence_list_label = QLabel("[Sentence List]")
        self.sentence_list_output = QTextEdit(self)
        self.sentence_list_output.setFont(self.ui_helper.small_font)
        self.sentence_list_output.setReadOnly(True)

        # QLabel initialization (FPS display)
        self.overall_fps_label = QLabel("FPS: N/A", self)
        self.overall_fps_label.setAlignment(Qt.AlignRight)

        # QPushButton initialization (start and stop video)
        self.resume_button = QPushButton("Resume", self)
        self.pause_button = QPushButton("Pause", self)

        # QPushButton initialization (text input and delete)
        self.add_button = QPushButton("Add", self)
        self.del_button = QPushButton("Delete", self)
        self.clear_button = QPushButton("Clear", self)

        # Connect button events
        self.resume_button.clicked.connect(self.resume)
        self.pause_button.clicked.connect(self.pause)
        self.add_button.clicked.connect(self.push_sentence)
        self.del_button.clicked.connect(self.pop_sentence)
        self.clear_button.clicked.connect(self.clear_text_list)

        # Connect sentence input enter key event
        self.sentence_input.returnPressed.connect(self.push_sentence)  # enter key

        # Connect sentence input updated event
        self.__view_model.get_sentence_list_updated_signal().connect(self.refresh_sentence_list)

        self.layout_setup()

        self.video_worker_list: List[VideoWorker] = []
        self.video_worker_setup()

        self.start()
        self.refresh_sentence_list()

    def video_worker_setup(self):
        for channel_idx in range(self.ui_config.num_channels):
            sentence_list_updated_signal = self.__view_model.get_sentence_list_updated_signal()
            video_producer = VideoProducer(
                channel_idx,
                self.base_path,
                self.adjusted_video_path_lists[channel_idx],
                self.ui_helper.video_size,
                sentence_list_updated_signal,
            )

            [scaled_video_frame_updated_signal, origin_video_frame_updated_signal,
             video_source_changed_signal] = video_producer.get_video_frame_updated_signal()
            scaled_video_frame_updated_signal.connect(self.update_scaled_video_frame)

            video_consumer = ClipVideoConsumer(channel_idx, self.ui_config.number_of_alarms,
                                               origin_video_frame_updated_signal, video_source_changed_signal,
                                               sentence_list_updated_signal,
                                               self.__view_model)

            video_consumer.get_update_each_fps_signal().connect(self.update_each_fps)
            video_consumer.get_update_overall_fps_signal().connect(self.update_overall_fps)
            video_consumer.get_update_text_output_signal().connect(self.update_text_output)
            video_consumer.get_clear_text_output_signal().connect(self.clear_text_output)

            video_worker = VideoWorker(channel_idx, video_producer, video_consumer)
            self.video_worker_list.append(video_worker)


    def layout_setup(self):
        video_layout = QVBoxLayout()
        video_box = self.generate_video_box()
        video_layout.addLayout(video_box)

        control_layout = QVBoxLayout()
        button_box = self.generate_control_ui()
        control_layout.addLayout(button_box)

        if self.ui_helper.ui_config.terminal_mode:
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
        terminal_box.addWidget(self.sentence_input_label)
        input_box = QHBoxLayout()
        input_box.addWidget(self.sentence_input)
        input_box.addWidget(self.add_button)
        input_box.addWidget(self.del_button)
        input_box.addWidget(self.clear_button)
        terminal_box.addLayout(input_box)
        terminal_box.addWidget(self.sentence_list_label)
        terminal_box.addWidget(self.sentence_list_output)
        return terminal_box

    def generate_control_ui(self):
        # [resume] or [pause] | [fps info] | [exit]
        control_box = QHBoxLayout()
        control_box.addWidget(self.resume_button)
        control_box.addWidget(self.pause_button)
        control_box.addWidget(self.overall_fps_label)
        if self.ui_helper.ui_config.fullscreen_mode:
            exit_button = QPushButton("Exit", self)
            exit_button.clicked.connect(self.close_application)
            control_box.addWidget(exit_button)
        return control_box

    def generate_video_box(self):
        if self.ui_helper.ui_config.num_channels == 1:
            # case of single-channel
            video_layout = QVBoxLayout()

            if self.ui_helper.ui_config.show_each_fps_label:
                each_fps_label = QLabel("", self)
                each_fps_label.setFont(self.ui_helper.small_font)
                each_fps_label.setFixedHeight(self.ui_helper.small_font_line_height)
                each_fps_label.setAlignment(Qt.AlignRight)
                self.each_fps_label_list.append(each_fps_label)
                video_layout.addWidget(each_fps_label)
            self.each_fps_info_list.append({"dxnn_fps": -1, "sol_fps": -1})

            video_label = QLabel(self)
            video_label.setAlignment(Qt.AlignCenter)
            self.video_label_list.append(video_label)
            video_layout.addWidget(video_label)

            text_output = QTextEdit(self)
            text_output.setFont(self.ui_helper.large_font)
            text_output.setFixedHeight(self.ui_helper.large_font_line_height * self.ui_helper.ui_config.number_of_alarms)
            text_output.setAlignment(Qt.AlignCenter)
            text_output.setReadOnly(True)
            self.text_output_list.append(text_output)
            video_layout.addWidget(text_output)

            return video_layout
        else:
            # case of multi-channel
            video_grid_layout = QGridLayout()

            for i in range(self.ui_helper.ui_config.num_channels):

                video_layout = QVBoxLayout()

                if self.ui_helper.ui_config.show_each_fps_label:
                    each_fps_label = QLabel("FPS: N/A", self)
                    each_fps_label.setFont(self.ui_helper.smaller_font)
                    each_fps_label.setFixedHeight(self.ui_helper.smaller_font_line_height)
                    each_fps_label.setAlignment(Qt.AlignRight)
                    self.each_fps_label_list.append(each_fps_label)
                    video_layout.addWidget(each_fps_label)
                self.each_fps_info_list.append({"dxnn_fps": -1, "sol_fps": -1})     # for calculate overall fps

                video_label = QLabel(self)
                video_label.setAlignment(Qt.AlignCenter)
                video_layout.addWidget(video_label)

                text_output = QTextEdit(self)
                text_output.setFont(self.ui_helper.small_font)
                text_output.setFixedWidth(self.ui_helper.video_size[0])
                text_output.setFixedHeight(self.ui_helper.small_font_line_height * self.ui_helper.ui_config.number_of_alarms)
                text_output.setAlignment(Qt.AlignCenter)
                text_output.setReadOnly(True)
                video_layout.addWidget(text_output)

                video_grid_layout.addLayout(video_layout, i // self.ui_helper.grid_cols, i % self.ui_helper.grid_cols)

                self.text_output_list.append(text_output)
                self.video_label_list.append(video_label)

            return video_grid_layout

    def start(self):
        for video_worker in self.video_worker_list:
            video_worker.get_video_producer().start()
            video_worker.get_video_consumer().start()

        self.resume_button.hide()

    def resume(self):
        self.resume_button.hide()

        for video_worker in self.video_worker_list:
            video_worker.get_video_producer().resume()
            video_worker.get_video_consumer().resume()

        self.pause_button.show()

    def pause(self):
        self.pause_button.hide()

        for video_worker in self.video_worker_list:
            video_worker.get_video_producer().pause()
            video_worker.get_video_consumer().pause()

        self.resume_button.show()

    def update_scaled_video_frame(self, channel_idx: int, qt_img):
        self.video_label_list[channel_idx].setPixmap(QPixmap.fromImage(qt_img))

    def clear_text_output(self, idx: int):
        self.text_output_list[idx].clear()

    def update_text_output(self, idx: int, text: str, progress: int, score: float):
        prefix_str = ""
        if self.ui_config.show_percent:
            prefix_str += "[" + str(progress) + "%]"

        if self.ui_config.show_score:
            prefix_str +=  "[" + str(round(score, 4)) + "]"

        self.text_output_list[idx].append(prefix_str + text)

    def show_toast(self, text, title="Info", duration=3000, preset=ToastPreset.WARNING, position=ToastPosition.CENTER):
        toast = Toast(self)
        toast.setDuration(duration)
        toast.setTitle(title)
        toast.setText(text)
        toast.applyPreset(preset)
        Toast.setPosition(position)
        toast.show()

    def refresh_sentence_list(self):
        sentence_list = self.__view_model.get_sentence_list()
        self.sentence_list_output.clear()
        for item in sentence_list:
            self.sentence_list_output.append(item)

    def push_sentence(self):
        if len(self.sentence_input.text()) == 0:
            self.show_toast("Please enter a sentence and press the Add button.")
            return

        self.__view_model.push_sentence(self.sentence_input.text())

    def pop_sentence(self):
        if len(self.sentence_list_output.toPlainText()) == 0:
            self.show_toast("No sentences to delete.")
            return

        self.__view_model.pop_sentence()

    def clear_text_list(self):
        if len(self.sentence_list_output.toPlainText()) == 0:
            self.show_toast("No sentences to delete.")
            return

        self.__view_model.clear_sentence()

    def update_overall_fps(self):
        with self.__fps_lock:
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
        with self.__fps_lock:
            if self.ui_helper.ui_config.show_each_fps_label:
                self.each_fps_label_list[thread_idx].setText(f" NPU FPS: {dxnn_fps:.2f}, Render FPS: {sol_fps:.2f} ")

            self.each_fps_info_list[thread_idx]["dxnn_fps"] = dxnn_fps
            self.each_fps_info_list[thread_idx]["sol_fps"] = sol_fps

    @overrides
    def closeEvent(self, event):
        for video_worker in self.video_worker_list:
            video_worker.get_video_producer().stop()
            video_worker.get_video_consumer().stop()

        super().closeEvent(event)

    def close_application(self):
        self.close()
