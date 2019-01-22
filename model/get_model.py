from model.model.LSTM import LSTM
from model.model.TextCNN import TextCNN
from model.model.SFKS.BasicCNN import BasicCNN
from model.model.SFKS.co_matching import CoMatching
from model.model.SFKS.SeaReader.SeaReader import SeaReader
from model.model.SFKS.co_matching import CoMatching, CoMatching2, CoMatching3
from model.model.SFKS.multi_matching import MultiMatchNet
from model.model.SFKS.bert import SFKSBert
from model.model.bert import Bert

model_list = {
    "LSTM": LSTM,
    "TextCNN": TextCNN,
    "SFKSCNN": BasicCNN,
    "Comatching": CoMatching,
    "SeaReader": SeaReader,
    "Comatching2": CoMatching2,
    "Comatching3": CoMatching3,
    "MultiMatch": MultiMatchNet,
    "Bert": Bert,
    "SFKS_bert": SFKSBert
}


def get_model(name, config):
    if name in model_list.keys():
        return model_list[name](config)
    else:
        raise NotImplementedError
