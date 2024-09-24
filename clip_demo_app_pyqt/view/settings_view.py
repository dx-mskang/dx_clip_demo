from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QLineEdit, QSpinBox, QCheckBox, QPushButton, QWidget

from clip_demo_app_pyqt.common.config.ui_config import UIConfig
from clip_demo_app_pyqt.data.input_data import InputData


class SettingsView(QMainWindow):
    def __init__(self, args, input_data: InputData, success_cb):
        super().__init__()
        self.model = None
        self.view_model = None
        self.main_app = None

        # CLI arguments
        self.base_path = args.base_path
        self.max_words = args.max_words
        self.feature_framerate = args.feature_framerate
        self.slice_framepos = args.slice_framepos

        # app config setting
        self.number_of_channels = args.number_of_channels
        self.terminal_mode = args.terminal_mode
        self.camera_mode = args.camera_mode

        # input data setting
        self.video_path_lists = input_data.video_path_lists
        self.sentence_alarm_threshold_list = input_data.sentence_alarm_threshold_list
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

        # Show Score checkbox
        self.show_score_checkbox = QCheckBox("Display Score", self)
        self.show_score_checkbox.setChecked(self.ui_config.show_score)
        layout.addWidget(self.show_score_checkbox)

        # Show FPS Label checkbox
        self.show_each_fps_label_checkbox = QCheckBox("Display FPS for each video", self)
        self.show_each_fps_label_checkbox.setChecked(self.ui_config.show_each_fps_label)
        layout.addWidget(self.show_each_fps_label_checkbox)

        # Terminal Mode checkbox
        self.terminal_mode_checkbox = QCheckBox("Terminal Mode", self)
        self.terminal_mode_checkbox.setChecked(self.terminal_mode)
        layout.addWidget(self.terminal_mode_checkbox)

        # Camera Mode checkbox
        self.camera_mode_checkbox = QCheckBox("Camera Mode", self)
        self.camera_mode_checkbox.setChecked(self.camera_mode)
        layout.addWidget(self.camera_mode_checkbox)

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

        # app config (using args)
        self.ui_config.num_channels = self.number_of_channels_input.value()
        self.ui_config.terminal_mode = self.terminal_mode_checkbox.isChecked()
        self.ui_config.camera_mode = self.camera_mode_checkbox.isChecked()

        # add config (hard-coded)
        self.ui_config.show_percent = self.show_percent_checkbox.isChecked()
        self.ui_config.show_score = self.show_score_checkbox.isChecked()
        self.ui_config.show_each_fps_label = self.show_each_fps_label_checkbox.isChecked()
        self.ui_config.fullscreen_mode = self.fullscreen_mode_checkbox.isChecked()
        self.ui_config.dark_theme = self.dark_theme_checkbox.isChecked()

        # adjust_video_path_lists and num_channels
        [self.adjusted_video_path_lists, self.ui_config.num_channels] = self.get_adjust_video_path_lists(
            self.video_path_lists, self.ui_config.num_channels, self.ui_config.camera_mode)

        # Close the settings window and start the main app
        self.close()
        self.start_main_app()

    def start_main_app(self):
        self.success_cb(self)

    @staticmethod
    def get_adjust_video_path_lists(video_path_lists, number_of_channels, camera_mode):
        result_num_of_channels = number_of_channels

        if number_of_channels <= 1:
            video_path_lists_for_single = []
            enriched_path_list = []
            for path_list in video_path_lists:
                for path in path_list:
                    enriched_path_list.append(path)
            video_path_lists_for_single.append(enriched_path_list)
            result_video_path_lists = video_path_lists_for_single

            if camera_mode:
                result_video_path_lists[0] = ["/dev/video0"]
        else:
            result_video_path_lists = video_path_lists[:number_of_channels]

            if camera_mode:
                result_video_path_lists.append(["/dev/video0"])
                result_num_of_channels = number_of_channels + 1

        return [result_video_path_lists, result_num_of_channels]
