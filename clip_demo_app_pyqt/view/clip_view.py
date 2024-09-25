import threading
from typing import List

import qdarkstyle
from PyQt5.QtCore import Qt, QObject
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMainWindow, QTextEdit, QLineEdit, \
    QHBoxLayout, QGridLayout, QDialog, QScrollArea
from overrides import overrides
from pyqttoast import ToastPreset, ToastPosition, Toast

from clip_demo_app_pyqt.common.base import CombinedMeta, Base
from clip_demo_app_pyqt.viewmodel.clip_view_model import ClipViewModel
from clip_demo_app_pyqt.common.config.ui_config import UIHelper, UIConfig
from clip_demo_app_pyqt.view.multithreading.clip_video_consumer import ClipVideoConsumer
from clip_demo_app_pyqt.view.multithreading.video_producer import VideoProducer
from clip_demo_app_pyqt.view.multithreading.video_worker import VideoWorker


class AddSentenceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Sentence")
        self.setModal(True)

        # QLineEdit and QTextEdit initialization (text input and display)
        self.sentence_input_label = QLabel("[Input Terminal]", self)
        self.sentence_input = QLineEdit(self)
        self.sentence_input.setMinimumWidth(parent.ui_config.sentence_input_min_width)
        self.sentence_input.setPlaceholderText("Please enter a sentence.")

        # Layout configuration
        terminal_box = QVBoxLayout()
        terminal_box.addWidget(self.sentence_input_label)
        terminal_box.addWidget(self.sentence_input)

        # Add Cancel and Submit buttons
        button_box = QHBoxLayout()
        cancel_button = QPushButton("Cancel", self)
        submit_button = QPushButton("Submit", self)
        button_box.addWidget(cancel_button)
        button_box.addWidget(submit_button)

        terminal_box.addLayout(button_box)
        self.setLayout(terminal_box)

        # Connect button actions
        cancel_button.clicked.connect(self.reject)
        submit_button.clicked.connect(self.accept)

        # Connect sentence input enter key event
        self.sentence_input.returnPressed.connect(self.accept)  # enter key

    def get_sentence(self):
        return self.sentence_input.text()


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
        self.text_output_layout_list: List[QVBoxLayout] = []

        self.sentence_list_label = QLabel("[Sentence List]")

        self.sentence_list_layout = QVBoxLayout()
        self.sentence_widget = QWidget()
        self.sentence_widget.setLayout(self.sentence_list_layout)

        self.sentence_list_scroll_area = QScrollArea()
        self.sentence_list_scroll_area.setWidget(self.sentence_widget)
        self.sentence_list_scroll_area.setWidgetResizable(True)
        self.sentence_list_scroll_area.setMinimumHeight(self.ui_config.sentence_list_scroll_area_min_height)
        self.sentence_list_scroll_area.setFixedWidth(self.ui_config.sentence_list_scroll_area_fixed_width)

        # QLabel initialization (FPS display)
        self.overall_fps_label = QLabel("FPS: N/A", self)
        self.overall_fps_label.setAlignment(Qt.AlignRight)

        # QPushButton initialization (start and stop video)
        self.resume_button = QPushButton("Resume", self)
        self.pause_button = QPushButton("Pause", self)

        # QPushButton initialization (text input and delete)
        self.add_button = QPushButton("Add", self)
        self.clear_button = QPushButton("Clear", self)

        # Connect button events
        self.resume_button.clicked.connect(self.resume)
        self.pause_button.clicked.connect(self.pause)
        self.add_button.clicked.connect(self.open_add_sentence_dialog)
        self.clear_button.clicked.connect(self.clear_text_list)

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
        # [Sentence List]
        # [add] | [clr]
        # --------------------------------
        # [sentence list area]
        # ...
        # --------------------------------
        terminal_box = QVBoxLayout()
        input_control_box = QHBoxLayout()
        input_control_box.addWidget(self.add_button)
        input_control_box.addWidget(self.clear_button)
        terminal_box.addLayout(input_control_box)
        terminal_box.addWidget(self.sentence_list_label)
        terminal_box.addWidget(self.sentence_list_scroll_area)
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

            text_output_layout = QVBoxLayout()
            text_output_widget = QWidget()
            text_output_widget.setLayout(text_output_layout)

            text_output_area = QScrollArea()
            text_output_area.setContentsMargins(0, 0, 0, 0)
            text_output_area.setWidget(text_output_widget)
            text_output_area.setWidgetResizable(True)
            text_output_area.setFixedHeight(self.ui_helper.large_font_line_height * self.ui_helper.ui_config.number_of_alarms + self.ui_helper.large_font_bottom_padding)
            text_output_area.setMinimumWidth(self.ui_helper.video_size[0])  # 너비 설정
            text_output_area.setAlignment(Qt.AlignCenter)
            text_output_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            text_output_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            video_layout.addWidget(text_output_area)
            self.text_output_layout_list.append(text_output_layout)

            return video_layout
        else:
            # case of multi-channel
            video_grid_layout = QGridLayout()

            for i in range(self.ui_helper.ui_config.num_channels):

                video_layout = QVBoxLayout()
                video_layout.setSpacing(0)

                video_label = QLabel(self)
                video_label.setAlignment(Qt.AlignCenter)
                # video_label.setStyleSheet("border: 1px solid yellow; padding: 0px;")
                video_label.setContentsMargins(0, 0, 0, 0)


                if self.ui_helper.ui_config.show_each_fps_label:
                    each_fps_label = QLabel("FPS: N/A", self)
                    each_fps_label.setFont(self.ui_helper.smaller_font)
                    each_fps_label.setFixedHeight(self.ui_helper.smaller_font_line_height)
                    each_fps_label.setAlignment(Qt.AlignRight)
                    self.each_fps_label_list.append(each_fps_label)
                    video_layout.addWidget(each_fps_label)
                self.each_fps_info_list.append({"dxnn_fps": -1, "sol_fps": -1})     # for calculate overall fps

                text_output_layout = QVBoxLayout()
                text_output_layout.setSpacing(0)
                text_output_layout.setContentsMargins(0, 0, 0, 0)
                text_output_widget = QWidget()
                text_output_widget.setLayout(text_output_layout)

                text_output_area = QScrollArea()
                text_output_area.setContentsMargins(0, 0, 0, 0)
                text_output_area.setWidget(text_output_widget)
                text_output_area.setWidgetResizable(True)
                text_output_area.setFixedHeight(
                    self.ui_helper.small_font_line_height * self.ui_helper.ui_config.number_of_alarms + self.ui_helper.small_font_bottom_padding)
                text_output_area.setMinimumWidth(self.ui_helper.video_size[0])  # 너비 설정
                text_output_area.setAlignment(Qt.AlignCenter)
                text_output_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                text_output_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

                video_layout.addWidget(video_label)
                video_layout.addWidget(text_output_area)

                video_grid_layout.addLayout(video_layout, i // self.ui_helper.grid_cols, i % self.ui_helper.grid_cols)

                self.text_output_layout_list.append(text_output_layout)
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
        text_output_layout = self.text_output_layout_list[idx]
        while text_output_layout.count():
            item = text_output_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                self.clear_layout(item.layout())

    def update_text_output(self, idx: int, text: str, progress: int, score: float):
        prefix_str = ""
        if self.ui_config.show_percent:
            prefix_str += "[" + str(progress) + "%]"

        if self.ui_config.show_score:
            prefix_str +=  "[" + str(round(score, 4)) + "]"

        # text_output box layout
        # [prefix]|[text]
        text_output_box = QHBoxLayout()
        text_output_box.setSpacing(0)
        text_output_box.setContentsMargins(0, 0, 0, 0)

        if self.ui_helper.ui_config.num_channels > 1:
            font = self.ui_helper.small_font
            line_height = self.ui_helper.small_font_line_height
            prefix_text_fixed_width = self.ui_helper.small_font_prefix_text_fixed_width
        else:
            font = self.ui_helper.large_font
            line_height = self.ui_helper.large_font_line_height
            prefix_text_fixed_width = self.ui_helper.large_font_prefix_text_fixed_width

        if prefix_str != "":
            # prefix
            prefix_label = QLabel(prefix_str, self)
            prefix_label.setFont(font)
            prefix_label.setFixedHeight(line_height)
            prefix_label.setFixedWidth(prefix_text_fixed_width)
            # prefix_label.setStyleSheet("border: 1px solid yellow; padding: 0px;")
            prefix_label.setContentsMargins(0, 0, 0, 0)
            text_output_box.addWidget(prefix_label)

        # text_output
        text_output_label = QLabel(text, self)
        text_output_label.setFont(font)
        text_output_label.setFixedHeight(line_height)
        # text_output_label.setMinimumWidth(self.ui_helper.video_size[0])  # 너비 설정
        # text_output_label.setStyleSheet("border: 1px solid yellow; padding: 0px;")
        text_output_label.setContentsMargins(0, 0, 0, 0)
        text_output_box.addWidget(text_output_label)

        self.text_output_layout_list[idx].addLayout(text_output_box)


    def show_toast(self, text, title="Info", duration=3000, preset=ToastPreset.WARNING, position=ToastPosition.CENTER):
        toast = Toast(self)
        toast.setDuration(duration)
        toast.setTitle(title)
        toast.setText(text)
        toast.applyPreset(preset)
        Toast.setPosition(position)
        toast.show()

    def open_add_sentence_dialog(self):
        dialog = AddSentenceDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            sentence = dialog.get_sentence()
            self.add_sentence(sentence)

    def refresh_sentence_list(self):
        while self.sentence_list_layout.count():
            item = self.sentence_list_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                self.clear_layout(item.layout())

        sentence_list = self.__view_model.get_sentence_list()

        for idx, sentence in enumerate(sentence_list):
            # sentence box layout
            # [del]|[sentence]
            sentence_box = QHBoxLayout()

            del_button = QPushButton("Del", self)
            del_button.setFixedWidth(50)
            del_button.clicked.connect(lambda _, i=idx: self.delete_sentence(i))
            sentence_box.addWidget(del_button)

            sentence_label = QLabel(sentence, self)
            sentence_box.addWidget(sentence_label)

            self.sentence_list_layout.addLayout(sentence_box)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                self.clear_layout(item.layout())

    def add_sentence(self, sentence):
        if len(sentence) == 0:
            self.show_toast("Please enter a sentence and press the Add button.")
            return

        self.__view_model.push_sentence(sentence)

    def delete_sentence(self, index):
        self.__view_model.pop_sentence(index)

    def clear_text_list(self):
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
