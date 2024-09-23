from model.sentence_model import SentenceModel
from model.video_model import VideoModel


class ClipModel(SentenceModel, VideoModel):
    def __init__(self, base_path: str, video_path_lists: list[list[str]], sentence_list: list, sentence_alarm_threshold_list):
        SentenceModel.__init__(self, sentence_list, sentence_alarm_threshold_list)
        VideoModel.__init__(self, base_path, video_path_lists)
