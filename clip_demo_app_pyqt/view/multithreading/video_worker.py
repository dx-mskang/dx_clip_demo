from clip_demo_app_pyqt.common.base import Base
from clip_demo_app_pyqt.view.multithreading.video_consumer import VideoConsumer
from clip_demo_app_pyqt.view.multithreading.video_producer import VideoProducer


class VideoWorker(Base):
    def __init__(self, channel_idx: int, video_producer: VideoProducer, video_consumer: VideoConsumer):
        super().__init__()
        self.__channel_idx = channel_idx
        self.__video_producer = video_producer
        self.__video_consumer = video_consumer

    def get_channel_idx(self)-> int:
        return self.__channel_idx

    def get_video_producer(self) -> VideoProducer:
        return self.__video_producer

    def get_video_consumer(self) -> VideoConsumer:
        return self.__video_consumer