import copy
import threading

from pyqttoast import ToastPosition

from clip_demo_app_pyqt.data.input_data import InputData
from clip_demo_app_pyqt.lib.clip.dx_text_encoder import TextVectorUtil
from clip_demo_app_pyqt.model.model import Model


class AlarmInfo:
    def __init__(self, enabled: bool = False, title: str = "ALARM", position: int = ToastPosition.CENTER.value,
                 color: str = "#E8B849"):
        self.__enabled = enabled
        self.__title = title
        self.__position = position
        self.__color = color

    def setAlarm(self, value: bool):
        self.__enabled = value

    def setTitle(self, title: str):
        self.__title = title

    def setPosition(self, position):
        self.__position = position

    def setColor(self, color):
        self.__color = color

    def getAlarm(self) -> bool:
        return self.__enabled

    def getTitle(self) -> str:
        return self.__title

    def getPosition(self) -> int:
        return self.__position

    def getColor(self) -> str:
        return self.__color

    def to_dict(self):
        return {
            "enabled": self.__enabled,
            "title": self.__title,
            "position": self.__position,
            "color": self.__color
        }


class Sentence:
    __disabled = False

    def __init__(self, text: str,
                 score_min: float,
                 score_max: float,
                 score_threshold: float,
                 alarm: bool,
                 alarm_title: str,
                 alarm_position: int,
                 alarm_color: str):
        self.__text = text
        self.__score_min = score_min
        self.__score_max = score_max
        self.__score_threshold = score_threshold
        self.__alarm_info = AlarmInfo(alarm, alarm_title, alarm_position, alarm_color)

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

    def set_alarm(self, val: bool):
        self.__alarm_info.setAlarm(val)

    def get_disabled(self):
        return self.__disabled

    def get_alarm(self):
        return self.__alarm_info.getAlarm()

    def get_alarm_title(self):
        return self.__alarm_info.getTitle()

    def get_alarm_info(self) -> AlarmInfo:
        return self.__alarm_info

    def to_dict(self):
        return {
            "text": self.__text,
            "min_score": self.__score_min,
            "max_score": self.__score_max,
            "threshold": self.__score_threshold,
            "disabled": self.__disabled,
            "alarm_info": self.__alarm_info.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict):
        alarm_info_dict = data.get("alarm_info", AlarmInfo().to_dict())
        instance = cls(
            text=data["text"],
            score_min=data["min_score"],
            score_max=data["max_score"],
            score_threshold=data["threshold"],
            alarm=alarm_info_dict["enabled"],
            alarm_title=alarm_info_dict["title"],
            alarm_position=alarm_info_dict["position"],
            alarm_color=alarm_info_dict["color"],
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

    def get_alarm(self) -> bool:
        return self.__sentence.get_alarm()

    def get_alarm_title(self) -> str:
        return self.__sentence.get_alarm_info().getTitle()

    def get_alarm_position(self) -> int:
        return self.__sentence.get_alarm_info().getPosition()

    def get_alarm_color(self) -> str:
        return self.__sentence.get_alarm_info().getColor()


class SentenceModel(Model):
    def __init__(self, sentence_list: list[Sentence]):
        Model.__init__(self)
        self.__sentence_lock = threading.Lock()

        self.__sentence_list: list[Sentence] = sentence_list
        self.__sentence_vector_list: list = TextVectorUtil.get_text_vector_list(
            [sentence.get_text() for sentence in self.__sentence_list])

    def insert_sentence(self, text_input, score_min, score_max, score_threshold,
                        alarm, alarm_title, alarm_position, alarm_color, index=0):
        self.insert_sentence_obj(
            Sentence(text_input, score_min, score_max, score_threshold, alarm, alarm_title, alarm_position, alarm_color), index)

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

    def toggle_alarm(self, index):
        sentence = self.pop_sentence(index)
        sentence.set_alarm(not sentence.get_alarm())
        InputData().save_data()
        self.insert_sentence_obj(sentence, index)

    def update_sentence(self, text_input, score_min, score_max, score_threshold,
                        alarm, alarm_title, alarm_position, alarm_color, index):
        self.pop_sentence(index)
        self.insert_sentence(text_input, score_min, score_max, score_threshold,
                             alarm, alarm_title, alarm_position, alarm_color, index)

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
