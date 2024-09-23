import threading

from PyQt5.QtCore import QMutex

from clip_demo_app_pyqt.lib.clip.dx_text_encoder import SentenceVectorUtil
from clip_demo_app_pyqt.model.model import Model


class SentenceModel(Model):
    def __init__(self, sentence_list: list, sentence_alarm_threshold_list: list):
        Model.__init__(self)
        self.mutex = QMutex()
        # self.__sentence_lock = threading.Lock()

        self.__sentence_list = sentence_list
        self.__sentence_alarm_threshold_list = sentence_alarm_threshold_list

        # text embedding vector 미리 준비
        # with self.__sentence_lock:
        #     self.sentence_vector_list = SentenceVectorUtil.get_sentence_vector_list(self.__sentence_list)
        self.mutex.lock()
        try:
            self.sentence_vector_list = SentenceVectorUtil.get_sentence_vector_list(self.__sentence_list)
        finally:
            self.mutex.unlock()


    def push_sentence(self, text_input):
        # with self.__sentence_lock:
        self.mutex.lock()
        try:
            self.__sentence_list.append(text_input)
            self.sentence_vector_list = SentenceVectorUtil.get_sentence_vector_list(self.__sentence_list)
            self.__sentence_alarm_threshold_list.append([0.23, 0.31, 0.26])
        finally:
            self.mutex.unlock()

    def pop_sentence(self):
        # with self.__sentence_lock:
        self.mutex.lock()
        try:
            self.__sentence_list.pop(-1)
            self.sentence_vector_list.pop(-1)
            self.__sentence_alarm_threshold_list.pop(-1)
        finally:
            self.mutex.unlock()


    def clear_sentence(self):
        # with self.__sentence_lock:
        self.mutex.lock()
        try:
            self.__sentence_list.clear()
            self.sentence_vector_list.clear()
            self.__sentence_alarm_threshold_list.clear()
        finally:
            self.mutex.unlock()

    def get_sentence_lock(self):
        # return self.__sentence_lock
        return self.mutex

    def get_sentence_list(self) -> list:
        return self.__sentence_list

    def get_sentence_alarm_threshold_list(self) -> list:
        return self.__sentence_alarm_threshold_list