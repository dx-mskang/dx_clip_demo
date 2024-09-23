
import sys

from PyQt5.QtWidgets import QApplication

from clip_demo_app_pyqt.common.parser.parser_util import ParserUtil
from clip_demo_app_pyqt.data.input_data import InputData

# fmt: on

def main():
    # Get Input Arguments
    args = ParserUtil.get_args()

    app = QApplication(sys.argv)

    def success_cb(settings_ctx):
        # Run Main Windows
        from clip_demo_app_pyqt.model.clip_model import ClipModel
        from clip_demo_app_pyqt.view.clip_view import ClipView
        from clip_demo_app_pyqt.viewmodel.clip_view_model import ClipViewModel
        settings_ctx.model = ClipModel(settings_ctx.base_path, settings_ctx.adjusted_video_path_lists,
                                       settings_ctx.sentence_list, settings_ctx.sentence_alarm_threshold)
        settings_ctx.view_model = ClipViewModel(settings_ctx.model)
        settings_ctx.main_app = ClipView(settings_ctx.view_model, settings_ctx.ui_config,
                                         settings_ctx.base_path, settings_ctx.adjusted_video_path_lists)
        settings_ctx.main_app.show()

    # Run Setting Window
    from clip_demo_app_pyqt.view.settings_view import SettingsView

    input_data = InputData()
    settings_window = SettingsView(args, input_data, success_cb)
    settings_window.show()

    app_ret = app.exec_()

    sys.exit(app_ret)


if __name__ == "__main__":
    main()
