import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, ROUND_HALF_UP
from typing import List

import qdarkstyle
from PyQt5.QtCore import Qt, QObject, QEvent
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMainWindow, QLineEdit, \
    QHBoxLayout, QGridLayout, QDialog, QScrollArea, QDoubleSpinBox, QGroupBox, QCheckBox
from overrides import overrides
from pyqttoast import ToastPreset, ToastPosition, Toast

from clip_demo_app_pyqt.common.base import CombinedMeta, Base
from clip_demo_app_pyqt.data.input_data import InputData
from clip_demo_app_pyqt.model.sentence_model import Sentence
from clip_demo_app_pyqt.view.settings_view import MergedVideoGridInfo
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

        # sentence input
        sentence_input_label = QLabel("[Input Sentence]", self)
        self.sentence_input = QLineEdit(self)
        self.sentence_input.setMinimumWidth(parent.ui_config.sentence_input_min_width)
        self.sentence_input.setPlaceholderText("Please enter a sentence.")

        score_settings_label = QLabel("[Score Settings]", self)

        # score settings input
        decimals = UIConfig.score_settings_decimals
        single_step = UIConfig.score_settings_single_step

        score_settings_box = QHBoxLayout()
        score_min_label = QLabel("Min:", self)
        input_data = InputData()
        self.score_min_input = QDoubleSpinBox(self)
        self.score_min_input.setDecimals(decimals)
        self.score_min_input.setValue(input_data.get_default_sentence_score_min())
        self.score_min_input.setSingleStep(single_step)

        score_max_label = QLabel("Max:", self)
        self.score_max_input = QDoubleSpinBox(self)
        self.score_max_input.setDecimals(decimals)
        self.score_max_input.setValue(input_data.get_default_sentence_score_max())
        self.score_max_input.setSingleStep(single_step)

        score_threshold_label = QLabel("Threshold:", self)
        self.score_threshold_input = QDoubleSpinBox(self)
        self.score_threshold_input.setDecimals(decimals)
        self.score_threshold_input.setValue(input_data.get_default_sentence_score_threshold())
        self.score_threshold_input.setSingleStep(single_step)

        score_settings_box.addWidget(score_min_label)
        score_settings_box.addWidget(self.score_min_input)
        score_settings_box.addWidget(score_max_label)
        score_settings_box.addWidget(self.score_max_input)
        score_settings_box.addWidget(score_threshold_label)
        score_settings_box.addWidget(self.score_threshold_input)

        # Layout configuration
        settings_box = QVBoxLayout()
        settings_box.addWidget(sentence_input_label)
        settings_box.addWidget(self.sentence_input)
        settings_box.addWidget(score_settings_label)
        settings_box.addLayout(score_settings_box)

        # Add Cancel and Submit buttons
        button_box = QHBoxLayout()
        cancel_button = QPushButton("Cancel", self)
        submit_button = QPushButton("Submit", self)
        button_box.addWidget(cancel_button)
        button_box.addWidget(submit_button)

        settings_box.addLayout(button_box)
        self.setLayout(settings_box)

        # Connect button actions
        cancel_button.clicked.connect(self.reject)
        submit_button.clicked.connect(self.accept)

        # Connect sentence input enter key event
        self.sentence_input.returnPressed.connect(self.accept)  # enter key
        self.score_min_input.installEventFilter(self)
        self.score_max_input.installEventFilter(self)
        self.score_threshold_input.installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                self.accept()
                return True
        return super().eventFilter(obj, event)

    def get_sentence_input_text(self):
        return self.sentence_input.text()

    def get_score_min_input_value(self):
        return self.score_min_input.value()

    def get_score_max_input_value(self):
        return self.score_max_input.value()

    def get_score_threshold_input_value(self):
        return self.score_threshold_input.value()


class ModifySentenceDialog(AddSentenceDialog):
    def __init__(self, sentence, score_min, score_max, score_threshold, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Modify Sentence")

        # set previous sentence data
        self.sentence_input.setText(sentence.get_text())
        self.score_min_input.setValue(score_min)
        self.score_max_input.setValue(score_max)
        self.score_threshold_input.setValue(score_threshold)


class ClipView(Base, QMainWindow, metaclass=CombinedMeta):
    def __init__(self, view_model: ClipViewModel, ui_config: UIConfig, base_path, adjusted_video_path_lists,
                 adjusted_video_grid_info):
        QMainWindow.__init__(self)
        QObject.__init__(self)
        self.root_layout = None
        self.prev_app_layout_container = None
        self.prev_app_layout = None
        self.__fps_lock = threading.Lock()
        self.__view_model = view_model

        self.base_path = base_path
        self.adjusted_video_path_lists = adjusted_video_path_lists
        self.adjusted_video_grid_info: MergedVideoGridInfo = adjusted_video_grid_info
        self.ui_config = ui_config

        self.ui_helper = UIHelper(self, self.ui_config)

        self.setWindowTitle("Video Processing App")

        if self.ui_config.dark_theme:
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        # List for calculating overall FPS
        self.each_fps_info_list = []

        # Widget initialization
        self.each_fps_label_list: List[QLabel] = []
        self.video_label_list: List[QLabel] = []
        self.sentence_output_layout_list: List[QVBoxLayout] = []

        self.sentence_list_label = QLabel("[Sentence List]")

        self.sentence_list_layout = QVBoxLayout()
        self.sentence_widget = QWidget()
        self.sentence_widget.setLayout(self.sentence_list_layout)

        self.sentence_list_scroll_area = QScrollArea()
        self.sentence_list_scroll_area.setWidget(self.sentence_widget)
        self.sentence_list_scroll_area.setWidgetResizable(True)
        self.sentence_list_scroll_area.setMinimumHeight(self.ui_config.sentence_list_scroll_area_min_height)
        self.sentence_list_scroll_area.setMinimumWidth(self.ui_config.sentence_list_scroll_area_min_width)
                                            
        # QLabel initialization (FPS display)
        self.overall_fps_label = QLabel("FPS: N/A", self)
        self.overall_fps_label.setAlignment(Qt.AlignRight)

        # QPushButton initialization (start and stop video)
        self.show_settings_button = QPushButton("Settings", self)
        self.resume_button = QPushButton("Resume", self)
        self.pause_button = QPushButton("Pause", self)

        self.show_settings_button.setCheckable(True)
        self.show_settings_button.setChecked(self.ui_config.settings_mode)

        self.show_percentage_button = QPushButton("Percent", self)
        self.show_percentage_button.setCheckable(True)
        self.show_percentage_button.setChecked(self.ui_config.show_percent)

        self.show_score_button = QPushButton("Score", self)
        self.show_score_button.setCheckable(True)
        self.show_score_button.setChecked(self.ui_config.show_score)

        # QPushButton initialization (text add and delete)
        self.add_button = QPushButton("Add", self)
        self.reset_button = QPushButton("Reset", self)
        self.clear_button = QPushButton("Clear", self)

        # Connect button events
        self.resume_button.clicked.connect(self.resume)
        self.pause_button.clicked.connect(self.pause)
        self.show_settings_button.clicked.connect(self.__toggle_settings)
        self.show_percentage_button.clicked.connect(self.__toggle_percent)
        self.show_score_button.clicked.connect(self.__toggle_score)

        self.add_button.clicked.connect(self.open_add_sentence_dialog)
        self.reset_button.clicked.connect(self.reset_text_list)
        self.clear_button.clicked.connect(self.clear_text_list)

        # Connect sentence input updated event
        self.__view_model.get_sentence_list_updated_signal().connect(self.refresh_sentence_list)

        self.__layout_setup()

        # Producer & Consumer config
        self.__producer_futures = []
        self.__consumer_futures = []
        self.__producer_thread_pool_executor = ThreadPoolExecutor(max_workers=self.ui_config.max_producer_worker)
        self.__consumer_thread_pool_executor = ThreadPoolExecutor(max_workers=self.ui_config.max_consumer_worker)

        self.__video_worker_list: List[VideoWorker] = []
        self.__running_video_worker = False
        self.__pause_video_worker = False
        self.__setup_video_worker_list()

        self.start()
        self.refresh_sentence_list()

        self.resize(self.ui_helper.video_area_w, self.ui_helper.video_area_h)

        if self.ui_config.fullscreen_mode:
            self.showFullScreen()

    @overrides()
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_video_size()

    def _refresh_video_size(self):
        if self.__video_worker_list:
            for video_worker in self.__video_worker_list:
                ratio = 0.90
                new_size = self.video_label_list[video_worker.get_channel_idx()].size()
                video_worker.get_video_producer().set_video_label_size(
                    (int(round(new_size.width() * ratio)), int(round(new_size.height() * ratio))))

    def __setup_video_worker_list(self):
        for channel_idx in range(self.ui_config.num_channels):
            sentence_list_updated_signal = self.__view_model.get_sentence_list_updated_signal()
            video_producer = VideoProducer(
                channel_idx,
                self.base_path,
                self.adjusted_video_path_lists[channel_idx],
                self.ui_helper.video_size,
                self.ui_config.producer_video_fps_sync_mode,
                self.ui_config.producer_video_frame_skip_interval,
                sentence_list_updated_signal,
            )

            [scaled_video_frame_updated_signal, video_source_changed_signal] = video_producer.get_video_frame_updated_signal()
            scaled_video_frame_updated_signal.connect(self.update_scaled_video_frame)

            video_consumer = ClipVideoConsumer(channel_idx, self.ui_config.number_of_alarms,
                                               video_source_changed_signal,
                                               sentence_list_updated_signal,
                                               self.ui_config.consumer_num_of_inference_per_sec,
                                               self.ui_config.consumer_max_np_array_similarity_queue,
                                               self.ui_config.consumer_video_fps_sync_mode,
                                               self.ui_config.inference_engine_async_mode,
                                               self.__view_model)

            video_consumer.get_update_each_fps_signal().connect(self.update_each_fps)
            video_consumer.get_update_overall_fps_signal().connect(self.update_overall_fps)
            video_consumer.get_update_sentence_output_signal().connect(self.update_sentence_output)
            video_consumer.get_clear_sentence_output_signal().connect(self.clear_sentence_output)

            video_worker = VideoWorker(channel_idx, video_producer, video_consumer)
            self.__video_worker_list.append(video_worker)

    def __layout_setup(self):
        # 루트 위젯과 레이아웃을 한 번만 생성
        if not hasattr(self, "root_widget"):
            self.settings_box_widget = QWidget()
            self.settings_box_layout = QVBoxLayout(self.settings_box_widget)
            self.settings_box_layout.addLayout(self.generate_settings_box())

            self.top_panel_layout = QVBoxLayout()
            self.top_panel_layout.addLayout(self.generate_control_ui())

            self.main_panel_layout = QHBoxLayout()
            self.main_panel_layout.addLayout(self.generate_video_box())
            self.main_panel_layout.addWidget(self.settings_box_widget)

            self.root_widget = QWidget()
            self.root_layout = QVBoxLayout(self.root_widget)
            self.root_layout.addLayout(self.top_panel_layout)
            self.root_layout.addLayout(self.main_panel_layout)

            self.setCentralWidget(self.root_widget)

        # settings_mode에 따라 settings_box의 표시 여부만 조정
        self.settings_box_widget.setVisible(self.ui_config.settings_mode)

    def __toggle_settings(self):
        # settings_mode 토글
        self.ui_config.settings_mode = not self.ui_config.settings_mode
        # 레이아웃 업데이트
        self.__layout_setup()

    def generate_settings_box(self):
        # settings layout
        # [Sentence List]
        # [add] | [clr]
        # --------------------------------
        # [sentence list area]
        # ...
        # --------------------------------
        settings_box = QVBoxLayout()
        input_control_box = QHBoxLayout()
        input_control_box.addWidget(self.add_button)
        input_control_box.addWidget(self.reset_button)
        input_control_box.addWidget(self.clear_button)
        settings_box.addLayout(input_control_box)
        settings_box.addWidget(self.sentence_list_label)
        settings_box.addWidget(self.sentence_list_scroll_area)
        return settings_box

    def generate_control_ui(self):
        # [resume] or [pause] | [fps info] | [exit]
        control_box = QHBoxLayout()

        ui_toggle_box = QGroupBox("UI Setting")
        ui_toggle_layout = QHBoxLayout()
        ui_toggle_layout.addWidget(self.show_percentage_button)
        ui_toggle_layout.addWidget(self.show_score_button)
        ui_toggle_box.setLayout(ui_toggle_layout)
        control_box.addWidget(ui_toggle_box)

        fps_box = QGroupBox("FPS info")
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(self.overall_fps_label)
        fps_box.setLayout(fps_layout)
        control_box.addWidget(fps_box)

        app_control_box = QGroupBox("App Control")
        app_control_layout = QHBoxLayout()
        app_control_layout.addWidget(self.show_settings_button)
        app_control_layout.addWidget(self.resume_button)
        app_control_layout.addWidget(self.pause_button)

        if self.ui_config.fullscreen_mode:
            exit_button = QPushButton("Exit", self)
            exit_button.clicked.connect(self.close_application)
            app_control_layout.addWidget(exit_button)

        app_control_box.setLayout(app_control_layout)
        control_box.addWidget(app_control_box)

        return control_box

    def generate_video_box(self):
        if self.ui_config.num_channels == 1:
            return self.__generate_single_channel_video_box()
        elif self.ui_config.num_channels >= 2:
            # case of multi-channel
            return self.__generate_multi_channel_video_box()
        else:
            raise RuntimeError("num_channels value is not correct")

    def __generate_single_channel_video_box(self):
        # case of single-channel
        [video_layout, sentence_output_layout, video_label] = self.__generate_video_box_impl(
            self.ui_helper.large_font, self.ui_helper.large_font_line_height, self.ui_helper.large_font_bottom_padding)
        self.video_label_list.append(video_label)
        self.sentence_output_layout_list.append(sentence_output_layout)
        return video_layout

    def __generate_multi_channel_video_box(self):
        if self.ui_config.merge_central_grid:
            return self.__generate_video_grid_layout_for_merge_central_grid()
        else:
            return self.__setup_video_grid_layout()

    def __generate_video_grid_layout_for_merge_central_grid(self):
        video_grid_layout = QGridLayout()
        grid = self.adjusted_video_grid_info

        outer_channel_idx = 0
        for row in range(grid.grid_size):
            for col in range(grid.grid_size):
                # Skip the central area
                if grid.center_start <= row < grid.center_end and grid.center_start <= col < grid.center_end:
                    continue

                if outer_channel_idx >= grid.outer_channel_max:
                    break

                # Create outer channel layout
                [video_layout, sentence_output_layout, video_label] = self.__generate_video_box_impl(
                    self.ui_helper.small_font, self.ui_helper.small_font_line_height,
                    self.ui_helper.small_font_bottom_padding)

                video_grid_layout.addLayout(video_layout, row, col)
                self.sentence_output_layout_list.append(sentence_output_layout)
                self.video_label_list.append(video_label)
                outer_channel_idx += 1

        # Central camera channel layout
        [merged_center_grid_layout, merged_center_grid_sentence_output_layout,
         merged_center_grid_label] = self.__generate_video_box_impl(
            self.ui_helper.large_font, self.ui_helper.large_font_line_height,
            self.ui_helper.large_font_bottom_padding, border_color="yellow")

        # Add camera layout to the central area
        video_grid_layout.addLayout(merged_center_grid_layout, grid.center_start, grid.center_start,
                                    grid.center_size, grid.center_size)
        self.sentence_output_layout_list.append(merged_center_grid_sentence_output_layout)
        self.video_label_list.append(merged_center_grid_label)
        return video_grid_layout

    def __setup_video_grid_layout(self):
        video_grid_layout = QGridLayout()
        for i in range(self.ui_config.num_channels):
            border_color = "gray"
            if i+1 == self.ui_config.num_channels and self.ui_config.camera_mode:
                border_color = "yellow"

            [video_layout, sentence_output_layout, video_label] = self.__generate_video_box_impl(
                self.ui_helper.small_font, self.ui_helper.small_font_line_height,
                self.ui_helper.small_font_bottom_padding, border_color=border_color)

            video_grid_layout.addLayout(video_layout, i // self.ui_helper.grid_cols, i % self.ui_helper.grid_cols)

            self.sentence_output_layout_list.append(sentence_output_layout)
            self.video_label_list.append(video_label)
        return video_grid_layout

    def __generate_video_box_impl(self, font, font_line_height, font_bottom_padding, border_color="gray"):
        video_layout = QVBoxLayout()
        video_layout.setSpacing(0)

        video_label = QLabel(self)
        video_label.setAlignment(Qt.AlignCenter)
        video_label.setStyleSheet("border: 1px solid " + border_color + "; padding: 0px; border-radius: 5px;")
        video_label.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(video_label)

        if self.ui_config.show_each_fps_label:
            each_fps_label = QLabel("FPS: N/A", self)
            each_fps_label.setFont(font)
            each_fps_label.setFixedHeight(font_line_height)
            each_fps_label.setAlignment(Qt.AlignRight)
            self.each_fps_label_list.append(each_fps_label)
            video_layout.addWidget(each_fps_label)
        self.each_fps_info_list.append({"dxnn_fps": -1, "sol_fps": -1})  # for calculate overall fps

        sentence_output_area, sentence_output_layout = self.__generate_sentence_output_area_impl(font_line_height, font_bottom_padding)
        video_layout.addWidget(sentence_output_area)
        return [video_layout, sentence_output_layout, video_label]

    def __generate_sentence_output_area_impl(self, font_line_height, font_bottom_padding):
        sentence_output_layout = QVBoxLayout()
        sentence_output_layout.setSpacing(0)
        sentence_output_layout.setContentsMargins(0, 0, 0, 0)
        sentence_output_widget = QWidget()
        sentence_output_widget.setLayout(sentence_output_layout)

        sentence_output_area = QScrollArea()
        sentence_output_area.setContentsMargins(0, 0, 0, 0)
        sentence_output_area.setWidget(sentence_output_widget)
        sentence_output_area.setWidgetResizable(True)
        sentence_output_area.setFixedHeight(
            font_line_height * self.ui_config.number_of_alarms + font_bottom_padding)
        sentence_output_area.setMinimumWidth(self.ui_helper.video_size[0])  # 너비 설정
        sentence_output_area.setAlignment(Qt.AlignCenter)
        sentence_output_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sentence_output_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return sentence_output_area, sentence_output_layout

    def worker_impl(self, channel_idx: int, worker_type: str, payload):
        video_worker = self.__video_worker_list[channel_idx]
        if worker_type == 'producer':
            payload = video_worker.get_video_producer().capture_frame()
            if payload is None:
                time.sleep(0.1)  # Adding a small sleep to avoid busy-waiting

            return payload
        elif worker_type == 'consumer':
            if payload is None:
                time.sleep(0.1)  # Adding a small sleep to avoid busy-waiting
                return channel_idx

            [channel_idx, frame, fps] = payload
            video_worker.get_video_consumer().process(channel_idx, frame, fps)
            return channel_idx
        else:
            raise RuntimeError('invalid type in worker_impl()')

    def start(self):
        self.__running_video_worker = True
        # start producer
        timer = threading.Timer(0.001, self.start_producer_worker)  # Thread separation and recursive repetition
        timer.start()

        # start consumer
        timer = threading.Timer(0.001, self.start_consumer_worker)  # Thread separation and recursive repetition
        timer.start()

        self.resume_button.hide()

    def start_producer_worker(self, worker_type='producer'):
        if self.__running_video_worker:
            if self.__pause_video_worker:
                timer = threading.Timer(0.1, self.start_producer_worker)  # Thread separation and recursive repetition
                timer.start()
                return

            video_channels = list(range(self.ui_config.num_channels))

            for channel_idx in video_channels:
                future = self.__producer_thread_pool_executor.submit(self.worker_impl, channel_idx, worker_type, None)
                self.__producer_futures.append(future)

            for future in as_completed(self.__producer_futures):
                try:
                    payload = future.result()
                    if payload is None:
                        break

                    logging.debug(payload)
                    [channel_idx, _, _] = payload
                    self.__video_worker_list[channel_idx].push_queue(payload)
                except Exception as e:
                    logging.error(f"Error processing video: {e}")

            self.__producer_futures.clear()

            timer = threading.Timer(0.001, self.start_producer_worker)
            timer.start()

    def start_consumer_worker(self, worker_type='consumer'):
        if self.__running_video_worker:
            if self.__pause_video_worker:
                timer = threading.Timer(0.1, self.start_consumer_worker)
                timer.start()
                return

            # if self.__consumer_queue.empty():
            for video_worker in self.__video_worker_list:
                payload = video_worker.pop_queue()
                if payload is not None:
                    [channel_idx, _, _] = payload
                    future = self.__consumer_thread_pool_executor.submit(self.worker_impl, channel_idx, worker_type, payload)
                    self.__consumer_futures.append(future)

            for future in as_completed(self.__consumer_futures):
                try:
                    channel_idx = future.result()
                    logging.debug("consumer worker task done, channel_idx: ", channel_idx)

                except Exception as e:
                    logging.error(f"Error processing video: {e}")

            # print("self.__consumer_futures clear() : ", len(self.__consumer_futures))
            self.__consumer_futures.clear()

            timer = threading.Timer(0.001, self.start_consumer_worker)    # Thread separation and recursive repetition
            timer.start()

    def resume(self):
        self.__pause_video_worker = False
        self.resume_button.hide()

        for video_worker in self.__video_worker_list:
            video_worker.get_video_producer().resume()
            video_worker.get_video_consumer().resume()

        self.pause_button.show()

    def pause(self):
        self.__pause_video_worker = True
        self.pause_button.hide()

        for video_worker in self.__video_worker_list:
            video_worker.get_video_producer().pause()
            video_worker.get_video_consumer().pause()

        self.resume_button.show()

    def __toggle_percent(self):
        self.ui_config.show_percent = self.show_percentage_button.isChecked()

    def __toggle_score(self):
        self.ui_config.show_score = self.show_score_button.isChecked()

    def update_scaled_video_frame(self, channel_idx: int, qt_img):
        self.video_label_list[channel_idx].setPixmap(QPixmap.fromImage(qt_img))

    def clear_sentence_output(self, idx: int):
        sentence_output_layout = self.sentence_output_layout_list[idx]
        while sentence_output_layout.count():
            item = sentence_output_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                self.clear_layout(item.layout())

    def update_sentence_output(self, idx: int, text: str, progress: int, score: float):
        prefix_str = ""
        if self.ui_config.show_percent:
            prefix_str += "[" + str(progress) + "%]"

        if self.ui_config.show_score:
            prefix_str += "[" + "{:.{}f}".format(score, self.ui_config.score_settings_decimals) + "]"

        # sentence_output box layout
        # [prefix]|[__text]
        sentence_output_box = QHBoxLayout()
        sentence_output_box.setSpacing(0)
        sentence_output_box.setContentsMargins(0, 0, 0, 0)

        is_camera_source = self.__video_worker_list[idx].get_video_producer().is_camera_source()
        is_single_channel = self.ui_config.num_channels == 1
        is_merged_center_grid = (self.adjusted_video_grid_info is not None) and (self.ui_config.num_channels == idx + 1)

        if is_camera_source or is_single_channel or is_merged_center_grid:
            font = self.ui_helper.large_font
            prefix_text_fixed_width = self.ui_helper.large_font_prefix_text_fixed_width
        else:
            font = self.ui_helper.small_font
            prefix_text_fixed_width = self.ui_helper.small_font_prefix_text_fixed_width

        if prefix_str != "":
            # prefix
            prefix_label = QLabel(prefix_str, self)
            prefix_label.setFont(font)
            prefix_label.setFixedWidth(prefix_text_fixed_width)
            # prefix_label.setStyleSheet("border: 1px solid red; padding: 0px;")
            prefix_label.setContentsMargins(0, 0, 0, 0)
            prefix_label.setAlignment(Qt.AlignTop)
            sentence_output_box.addWidget(prefix_label)

        # sentence_output
        sentence_output_label = QLabel(text, self)
        sentence_output_label.setFont(font)
        sentence_output_label.setMinimumWidth(self.ui_helper.video_size[0])  # 너비 설정
        # sentence_output_label.setStyleSheet("border: 1px solid green; padding: 0px;")
        sentence_output_label.setContentsMargins(0, 0, 0, 0)
        sentence_output_label.setAlignment(Qt.AlignTop)
        sentence_output_box.addWidget(sentence_output_label)

        self.sentence_output_layout_list[idx].addLayout(sentence_output_box)

    def show_toast(self, text, title="Info", duration=3000, preset=ToastPreset.WARNING, position=ToastPosition.CENTER):
        toast = Toast(self)
        toast.setDuration(duration)
        toast.setTitle(title)
        toast.setText(text)
        toast.applyPreset(preset)
        Toast.setPosition(position)
        toast.show()

    @staticmethod
    def __round_down_float(val: float, exp="0.000", rounding=ROUND_HALF_UP):
        return float(Decimal(val).quantize(Decimal(exp), rounding=rounding))

    def open_add_sentence_dialog(self):
        dialog = AddSentenceDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            sentence = dialog.get_sentence_input_text()
            score_min = self.__round_down_float(dialog.get_score_min_input_value())
            score_max = self.__round_down_float(dialog.get_score_max_input_value())
            score_threshold = self.__round_down_float(dialog.get_score_threshold_input_value())
            self.add_sentence(sentence, score_min, score_max, score_threshold)

    def open_modify_sentence_dialog(self, index):
        # 현재 문장 정보를 가져와 ModifySentenceDialog 열기
        sentence: Sentence = self.__view_model.get_sentence_list()[index]
        score_min = sentence.get_score_min()
        score_max = sentence.get_score_max()
        score_threshold = sentence.get_score_threshold()

        dialog = ModifySentenceDialog(sentence, score_min, score_max, score_threshold, self)
        if dialog.exec_() == QDialog.Accepted:
            updated_sentence = dialog.get_sentence_input_text()
            updated_min = self.__round_down_float(dialog.get_score_min_input_value())
            updated_max = self.__round_down_float(dialog.get_score_max_input_value())
            updated_threshold = self.__round_down_float(dialog.get_score_threshold_input_value())
            self.modify_sentence(updated_sentence, updated_min, updated_max, updated_threshold, index)

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

            mod_button = QPushButton("M", self)
            mod_button.setFixedWidth(20)
            mod_button.clicked.connect(lambda _, i=idx: self.open_modify_sentence_dialog(i))
            sentence_box.addWidget(mod_button)

            del_button = QPushButton("D", self)
            del_button.setFixedWidth(20)
            del_button.clicked.connect(lambda _, i=idx: self.delete_sentence(i))
            sentence_box.addWidget(del_button)

            sentence_toggle_checkbox = QCheckBox("", self)
            sentence_toggle_checkbox.setChecked(not sentence.get_disabled())
            sentence_toggle_checkbox.clicked.connect(lambda _, i=idx: self.toggle_sentence(i))
            sentence_toggle_checkbox.setFixedWidth(40)
            sentence_box.addWidget(sentence_toggle_checkbox)

            sentence_label = QLabel(sentence.get_text(), self)
            sentence_box.addWidget(sentence_label)

            self.sentence_list_layout.addLayout(sentence_box)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                self.clear_layout(item.layout())

    def add_sentence(self, sentence, score_min, score_max, score_threshold):
        if len(sentence) == 0:
            self.show_toast("Please enter a sentence and press the Add button.")
            return

        self.__view_model.insert_sentence(sentence, score_min, score_max, score_threshold)

    def delete_sentence(self, index):
        self.__view_model.pop_sentence(index)

    def toggle_sentence(self, index):
        self.__view_model.toggle_sentence(index)

    def modify_sentence(self, sentence, score_min, score_max, score_threshold, index):
        self.__view_model.update_sentence(sentence, score_min, score_max, score_threshold, index)

    def reset_text_list(self):
        self.__view_model.reset_sentence()

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
                    f" NPU FPS: {overall_dxnn_fps:.2f}, APP FPS: {overall_sol_fps:.2f} ")

    def update_each_fps(self, channel_idx, dxnn_fps, sol_fps):
        with self.__fps_lock:
            if self.ui_config.show_each_fps_label:
                self.each_fps_label_list[channel_idx].setText(f" NPU FPS: {dxnn_fps:.2f}, APP FPS: {sol_fps:.2f} ")

            self.each_fps_info_list[channel_idx]["dxnn_fps"] = dxnn_fps
            self.each_fps_info_list[channel_idx]["sol_fps"] = sol_fps

    @overrides
    def closeEvent(self, event):
        self.__running_video_worker = False

        for video_worker in self.__video_worker_list:
            video_worker.get_video_producer().stop()
            video_worker.get_video_consumer().stop()

        # self.__thread_pool_executor.shutdown(wait=True)
        self.__producer_thread_pool_executor.shutdown(wait=True)
        self.__consumer_thread_pool_executor.shutdown(wait=True)

        for future in self.__producer_futures:
            if not future.done():
                future.cancel()

        for future in self.__consumer_futures:
            if not future.done():
                future.cancel()

        super().closeEvent(event)

    def close_application(self):
        self.close()
