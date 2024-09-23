from model.clip_model import ClipModel
from viewmodel.sentence_view_model import SentenceViewModel
from viewmodel.video_view_model import VideoViewModel


class ClipViewModel(SentenceViewModel, VideoViewModel):
    def __init__(self, model: ClipModel):
        SentenceViewModel.__init__(self, model)
        VideoViewModel.__init__(self, model)
