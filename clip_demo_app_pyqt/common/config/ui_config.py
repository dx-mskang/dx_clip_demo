import math

from PyQt5.QtWidgets import QMainWindow, QApplication


class UIConfig:
    # Settings View
    num_channels = 16
    settings_mode = 0
    merge_central_grid = 0
    camera_mode = 0

    show_percent = False
    show_score = False
    alarm_only_on_camera_ch = True
    show_each_fps_label = False
    fullscreen_mode = True
    dark_theme = True

    # sentence(text) processing settings
    number_of_alarms = 2
    sentence_list_scroll_area_min_height = 150
    sentence_list_scroll_area_min_width = 450

    # video processing settings
    consumer_queue_size = 10
    consumer_num_of_inference_per_sec = 3
    consumer_max_np_array_similarity_queue = 5
    producer_video_fps_sync_mode = False
    consumer_video_fps_sync_mode = False
    producer_video_frame_skip_interval = 0  # 1st to 4th frames are processed, and the 5th frame is skipped when set to 5.
    max_producer_worker = 16
    max_consumer_worker = 4
    
    # Inference Engine RunAsync setting
    inference_engine_async_mode = True

    # Add Sentence Dialog
    sentence_input_min_width = 400
    score_settings_single_step = 0.005
    score_settings_decimals = 3
    alarm_title_input_min_width = 100

    # Clip View
    base_video_area_ratio = 0.65
    fullscreen_mode_negative_padding = 400

    # sentence alarm
    sentence_alarm_font_size = 26


class UIHelper:
    def __init__(self, ctx: QMainWindow, ui_config: UIConfig):
        super().__init__()
        self.ui_config = ui_config

        self.large_font = ctx.font()
        self.large_font.setPointSize(18)
        self.large_font_line_height = 44
        self.large_font_bottom_padding = 22
        self.large_font_prefix_text_fixed_width = 200

        self.small_font = ctx.font()
        self.small_font.setPointSize(9)
        self.small_font_line_height = 22
        self.small_font_bottom_padding = 11
        self.small_font_prefix_text_fixed_width = 100

        self.smaller_font = ctx.font()
        self.smaller_font.setPointSize(7)
        self.smaller_font_line_height = 16
        self.smaller_font_bottom_padding = 8

        screen_resolution = QApplication.desktop().screenGeometry()
        self.window_w = screen_resolution.width()
        self.window_h = screen_resolution.height()

        self.grid_rows, self.grid_cols = self.__calculate_grid_size(self.ui_config.num_channels)

        if not self.ui_config.fullscreen_mode:
            self.window_h -= self.ui_config.fullscreen_mode_negative_padding

        if self.ui_config.show_each_fps_label:
            self.window_h -= self.smaller_font_line_height * 4 * self.grid_rows

        print(f"Screen resolution: {self.window_w}x{self.window_h}")

        base_video_area_w = int(self.window_w * self.ui_config.base_video_area_ratio)  # Determine grid size considering the aspect ratio
        scale_factor = self.window_h / self.window_w
        calc_video_area_h = int(base_video_area_w * scale_factor)

        self.video_area_w = base_video_area_w
        self.video_area_h = calc_video_area_h
        self.video_size = (int(self.video_area_w / self.grid_rows), int(self.video_area_h / self.grid_cols))

    @staticmethod
    def __calculate_grid_size(num_channels):
        # Determine grid size considering the aspect ratio
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
            cols = math.ceil(math.sqrt(num_channels))  # Calculate the number of columns to form a nearly square shape
            rows = math.ceil(num_channels / cols)  # Calculate the number of rows by dividing the total number of channels by the number of columns
            return cols, rows
