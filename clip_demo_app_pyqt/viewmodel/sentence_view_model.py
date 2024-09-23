from PyQt5.QtCore import pyqtSignal

from clip_demo_app_pyqt.model.clip_model import SentenceModel
from clip_demo_app_pyqt.viewmodel.view_model import ViewModel


class SentenceViewModel(ViewModel):
    _sentence_list_updated_signal = pyqtSignal()

    def __init__(self, sentence_model: SentenceModel):
        super().__init__(sentence_model)
        self.__sentence_model = sentence_model
        # print("SentenceViewModel : " + str(dir(self)))

    def get_sentence_list_updated_signal(self):
        return self._sentence_list_updated_signal

    def push_sentence(self, sentence):
        self.__sentence_model.push_sentence(sentence)
        self._sentence_list_updated_signal.emit()

    def pop_sentence(self):
        self.__sentence_model.pop_sentence()
        self._sentence_list_updated_signal.emit()

    def clear_sentence(self):
        self.__sentence_model.clear_sentence()
        self._sentence_list_updated_signal.emit()

    def get_sentence_list(self) -> list:
        return self.__sentence_model.get_sentence_list()

    def get_sentence_vector_list(self) -> list:
        return self.__sentence_model.get_sentence_vector_list()

    def get_sentence_alarm_threshold_list(self) -> list:
        return self.__sentence_model.get_sentence_alarm_threshold_list()