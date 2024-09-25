import math

from PyQt5.QtWidgets import QMainWindow, QApplication

class UIConfig:
    num_channels = 16
    terminal_mode = 0
    camera_mode = 0

    show_percent = False
    show_score = False
    show_each_fps_label = False
    fullscreen_mode = True
    dark_theme = True
    number_of_alarms = 2
    sentence_input_min_width = 400
    sentence_list_scroll_area_min_height = 150
    sentence_list_scroll_area_fixed_width = 450
    video_consumer_queue_size = 1

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

        if not self.ui_config.fullscreen_mode:
            self.window_h -= 100

        print(f"Screen resolution: {self.window_w}x{self.window_h}")

        self.grid_rows, self.grid_cols = self.__calculate_grid_size(self.ui_config.num_channels)

        base_video_area_w = int(self.window_w * 0.65)  # 입력 터미널 UI 영역과 비디오 UI 영역의 비율 설정
        scale_factor = self.window_h / self.window_w
        calc_video_area_h = int(base_video_area_w * scale_factor)

        self.video_area_w = base_video_area_w
        self.video_area_h = calc_video_area_h
        self.video_size = (int(self.video_area_w / self.grid_rows), int(self.video_area_h / self.grid_cols))

    @staticmethod
    def __calculate_grid_size(num_channels):
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
            cols = math.ceil(math.sqrt(num_channels))  # 컬럼 수는 정사각형 형태에 가깝게 계산
            rows = math.ceil(num_channels / cols)  # 행 수는 총 채널 수를 컬럼 수로 나눈 값
            return cols, rows
