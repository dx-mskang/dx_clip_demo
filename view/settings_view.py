from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QLineEdit, QSpinBox, QCheckBox, QPushButton, QWidget

from common.config.ui_config import UIConfig
from data.input_data import InputData


class SettingsView(QMainWindow):
    def __init__(self, args, input_data: InputData, success_cb):
        super().__init__()
        self.model = None
        self.view_model = None
        self.main_app = None

        # CLI arguments
        self.base_path = args.base_path
        self.number_of_channels = args.number_of_channels
        self.max_words = args.max_words
        self.feature_framerate = args.feature_framerate
        self.slice_framepos = args.slice_framepos

        # input data setting
        self.video_path_lists = input_data.video_path_lists
        self.sentence_alarm_threshold = input_data.sentence_alarm_threshold
        self.sentence_list = input_data.sentence_list

        self.success_cb = success_cb

        self.adjusted_video_path_lists: list = []

        # UI config
        self.ui_config = UIConfig()

        self.setWindowTitle("Settings")
        self.setMinimumWidth(1000)

        # layout setup
        layout = QVBoxLayout()

        # features_path input
        self.features_path_label = QLabel("Features Path:")
        self.features_path_input = QLineEdit(self)
        self.features_path_input.setText(self.base_path)
        layout.addWidget(self.features_path_label)
        layout.addWidget(self.features_path_input)

        # number_of_channels input
        self.number_of_channels_label = QLabel("Number of Channels:")
        self.number_of_channels_input = QSpinBox(self)
        self.number_of_channels_input.setValue(self.number_of_channels)
        self.number_of_channels_input.setRange(1, 16)  # 1 ~ 16 채널로 제한
        layout.addWidget(self.number_of_channels_label)
        layout.addWidget(self.number_of_channels_input)

        # Show Percentage checkbox
        self.show_percent_checkbox = QCheckBox("Display Percentage", self)
        self.show_percent_checkbox.setChecked(self.ui_config.show_percent)
        layout.addWidget(self.show_percent_checkbox)

        # Show FPS Label checkbox
        self.show_each_fps_label_checkbox = QCheckBox("Display FPS for each video", self)
        self.show_each_fps_label_checkbox.setChecked(self.ui_config.show_each_fps_label)
        layout.addWidget(self.show_each_fps_label_checkbox)

        # Terminal Mode checkbox
        self.terminal_mode_checkbox = QCheckBox("Terminal Mode", self)
        self.terminal_mode_checkbox.setChecked(self.ui_config.terminal_mode)
        layout.addWidget(self.terminal_mode_checkbox)

        # Fullscreen Mode checkbox
        self.fullscreen_mode_checkbox = QCheckBox("Fullscreen Mode", self)
        self.fullscreen_mode_checkbox.setChecked(self.ui_config.fullscreen_mode)
        layout.addWidget(self.fullscreen_mode_checkbox)

        # Dark Theme checkbox
        self.dark_theme_checkbox = QCheckBox("Dark Theme", self)
        self.dark_theme_checkbox.setChecked(self.ui_config.dark_theme)
        layout.addWidget(self.dark_theme_checkbox)

        # Done button
        self.done_button = QPushButton("Done", self)
        self.done_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.done_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def apply_settings(self):
        # Apply the values entered by the user
        self.base_path = self.features_path_input.text()
        self.adjusted_video_path_lists = self.get_adjust_video_path_lists(self.video_path_lists,
                                                                          self.number_of_channels_input.value())
        self.ui_config.num_channels = self.number_of_channels_input.value()
        self.ui_config.show_percent = self.show_percent_checkbox.isChecked()
        self.ui_config.show_each_fps_label = self.show_each_fps_label_checkbox.isChecked()
        self.ui_config.terminal_mode = self.terminal_mode_checkbox.isChecked()
        self.ui_config.fullscreen_mode = self.fullscreen_mode_checkbox.isChecked()
        self.ui_config.dark_theme = self.dark_theme_checkbox.isChecked()

        # Close the settings window and start the main app
        self.close()
        self.start_main_app()

    def start_main_app(self):
        self.success_cb(self)

    @staticmethod
    def get_adjust_video_path_lists(gt_video_path_lists, number_of_channels):
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
