import copy
import threading

from clip_demo_app_pyqt.data.input_data import InputData
from clip_demo_app_pyqt.lib.clip.dx_text_encoder import TextVectorUtil
from clip_demo_app_pyqt.model.model import Model


class Sentence:
    __disabled = False

    def __init__(self, text: str,
                 score_min: float,
                 score_max: float,
                 score_threshold: float):
        self.__text = text
        self.__score_min = score_min
        self.__score_max = score_max
        self.__score_threshold = score_threshold

    def get_text(self) -> str:
        return self.__text

    def get_score_min(self) -> float:
        return self.__score_min

    def get_score_max(self) -> float:
        return self.__score_max

    def get_score_threshold(self) -> float:
        return self.__score_threshold

    def set_disabled(self, val: bool):
        self.__disabled = val

    def get_disabled(self):
        return self.__disabled

    def to_dict(self):
        return {
            "text": self.__text,
            "min_score": self.__score_min,
            "max_score": self.__score_max,
            "threshold": self.__score_threshold,
            "disabled": self.__disabled
        }

    @classmethod
    def from_dict(cls, data: dict):
        instance = cls(
            text=data["text"],
            score_min=data["min_score"],
            score_max=data["max_score"],
            score_threshold=data["threshold"]
        )
        instance.set_disabled(data.get("disabled", False))
        return instance


class SentenceOutput:
    def __init__(self, sentence: Sentence, score: float):
        self.__sentence = sentence
        self.__score = score
        self.__percentage = int(sentence.get_score_min() / sentence.get_score_max() * score)

        score_min = sentence.get_score_min()
        score_max = sentence.get_score_max()

        if score < score_min:
            percentage = 0
        elif score > score_max:
            percentage = 100
        else:
            percentage = int((score - score_min) / (score_max - score_min) * 100)

        self.__percentage = percentage

    def get_sentence_text(self) -> str:
        return self.__sentence.get_text()

    def get_score(self) -> float:
        return self.__score

    def get_percentage(self) -> int:
        return self.__percentage


class SentenceModel(Model):
    def __init__(self, sentence_list: list[Sentence]):
        Model.__init__(self)
        self.__sentence_lock = threading.Lock()

        self.__sentence_list: list[Sentence] = sentence_list
        self.__sentence_vector_list: list = TextVectorUtil.get_text_vector_list(
            [sentence.get_text() for sentence in self.__sentence_list])

    def insert_sentence(self, text_input,
                        score_min,
                        score_max,
                        score_threshold,
                        index=0):
        self.insert_sentence_obj(Sentence(text_input, score_min, score_max, score_threshold), index)

    def insert_sentence_obj(self, sentence: Sentence,
                            index=0):
        with self.__sentence_lock:
            self.__sentence_list.insert(index, sentence)
            self.__sentence_vector_list.insert(index, TextVectorUtil.get_text_vector(sentence.get_text()))
            InputData().save_data()

    def pop_sentence(self, index=None) -> Sentence:
        with self.__sentence_lock:
            if index is None:
                index = -1
            self.__sentence_vector_list.pop(index)
            sentence = self.__sentence_list.pop(index)
            InputData().save_data()
            return sentence

    def toggle_sentence(self, index):
        sentence = self.pop_sentence(index)
        sentence.set_disabled(not sentence.get_disabled())
        InputData().save_data()
        self.insert_sentence_obj(sentence, index)

    def update_sentence(self, text_input, score_min, score_max, score_threshold, index):
        self.pop_sentence(index)
        self.insert_sentence(text_input, score_min, score_max, score_threshold, index)

    def reset_sentence(self):
        with self.__sentence_lock:
            input_data = InputData()
            input_data.load_data(force=True)
            self.__sentence_list: list[Sentence] = input_data.get_sentence_list()
            self.__sentence_vector_list: list = TextVectorUtil.get_text_vector_list(
                [sentence.get_text() for sentence in self.__sentence_list])

    def clear_sentence(self):
        with self.__sentence_lock:
            self.__sentence_list.clear()
            self.__sentence_vector_list.clear()
            InputData().save_data()

    def get_sentence_list(self) -> list:
        with self.__sentence_lock:
            return copy.deepcopy(self.__sentence_list)

    def get_sentence_vector_list(self) -> list:
        with self.__sentence_lock:
            return copy.deepcopy(self.__sentence_vector_list)
