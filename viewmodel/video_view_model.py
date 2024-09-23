from model.clip_model import VideoModel
from viewmodel.view_model import ViewModel


class VideoViewModel(ViewModel):
    def __init__(self, video_model: VideoModel):
        super().__init__(video_model)
        self.__video_model = video_model
        self._num_channels = self.__video_model.get_num_channels()