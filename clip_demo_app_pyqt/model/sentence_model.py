import copy
import threading

from clip_demo_app_pyqt.lib.clip.dx_text_encoder import TextVectorUtil
from clip_demo_app_pyqt.model.model import Model

class Sentence:
    def __init__(self, text: str,
                 score_min: float,
                 score_max: float,
                 score_threshold: float):
        self.__text = text
        self.__score_min = score_min
        self.__score_max = score_max
        self.__score_threshold = score_threshold

    def getText(self) -> str:
        return self.__text

    def getScoreMin(self) -> float:
        return self.__score_min

    def getScoreMax(self) -> float:
        return self.__score_max

    def getScoreThreshold(self) -> float:
        return self.__score_threshold

class SentenceOutput:
    def __init__(self, sentence: Sentence, score: float):
        self.__sentence = sentence
        self.__score = score
        self.__percentage = int(sentence.getScoreMin() / sentence.getScoreMax() * score)

        score_min = sentence.getScoreMin()
        score_max = sentence.getScoreMax()

        if score < score_min:
            percentage = 0
        elif score > score_max:
            percentage = 100
        else:
            percentage = int((score - score_min) / (score_max - score_min) * 100)

        self.__percentage = percentage

    def getSentenceText(self) -> str:
        return self.__sentence.getText()

    def getScore(self) -> float:
        return self.__score

    def getPercentage(self) -> int:
        return self.__percentage


class SentenceModel(Model):
    def __init__(self, sentence_list: list[Sentence]):
        Model.__init__(self)
        self.__sentence_lock = threading.Lock()

        self.__sentence_list = sentence_list
        self.__sentence_vector_list  = TextVectorUtil.get_text_vector_list(
            [sentence.getText() for sentence in self.__sentence_list])

    def push_sentence(self, text_input,
                      score_min,
                      score_max,
                      score_threshold):
        with self.__sentence_lock:
            self.__sentence_list.append(Sentence(text_input, score_min, score_max, score_threshold))
            self.__sentence_vector_list.append(TextVectorUtil.get_text_vector(text_input))

    def pop_sentence(self, index=None):
        with self.__sentence_lock:
            if index is None:
                index = -1
            self.__sentence_list.pop(index)
            self.__sentence_vector_list.pop(index)

    def clear_sentence(self):
        with self.__sentence_lock:
            self.__sentence_list.clear()
            self.__sentence_vector_list.clear()

    def get_sentence_list(self) -> list:
        with self.__sentence_lock:
            return copy.deepcopy(self.__sentence_list)

    def get_sentence_vector_list(self) -> list:
        with self.__sentence_lock:
            return copy.deepcopy(self.__sentence_vector_list)

