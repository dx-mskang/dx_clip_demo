import copy
import threading

from clip_demo_app_pyqt.data.input_data import InputData
from clip_demo_app_pyqt.lib.clip.dx_text_encoder import SentenceVectorUtil
from clip_demo_app_pyqt.model.model import Model


class SentenceModel(Model):
    def __init__(self, sentence_list: list, sentence_alarm_threshold_list: list):
        Model.__init__(self)
        self.__sentence_lock = threading.Lock()

        self.__sentence_list = sentence_list
        self.__sentence_alarm_threshold_list = sentence_alarm_threshold_list
        self.__sentence_vector_list = SentenceVectorUtil.get_sentence_vector_list(self.__sentence_list)

    def push_sentence(self, text_input):
        with self.__sentence_lock:
            self.__sentence_list.append(text_input)
            self.__sentence_vector_list.append(SentenceVectorUtil.get_sentence_vector_list([text_input])[0])
            self.__sentence_alarm_threshold_list.append(InputData.default_sentence_threshold)

    def pop_sentence(self, index=None):
        with self.__sentence_lock:
            if index is None:
                index = -1
            self.__sentence_list.pop(index)
            self.__sentence_vector_list.pop(index)
            self.__sentence_alarm_threshold_list.pop(index)

    def clear_sentence(self):
        with self.__sentence_lock:
            self.__sentence_list.clear()
            self.__sentence_vector_list.clear()
            self.__sentence_alarm_threshold_list.clear()

    def get_sentence_list(self) -> list:
        with self.__sentence_lock:
            return copy.deepcopy(self.__sentence_list)

    def get_sentence_vector_list(self) -> list:
        with self.__sentence_lock:
            return copy.deepcopy(self.__sentence_vector_list)

    def get_sentence_alarm_threshold_list(self) -> list:
        with self.__sentence_lock:
            return copy.deepcopy(self.__sentence_alarm_threshold_list)
