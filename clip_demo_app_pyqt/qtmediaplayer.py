from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QUrl
import sys

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()
        self.setCentralWidget(videoWidget)
        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile("/home/inventec/deepx/dx_pia_demo/assets/media/ROSE_APT.mp4")))
        self.mediaPlayer.play()

app = QApplication(sys.argv)
player = VideoPlayer()
player.show()
sys.exit(app.exec_())


