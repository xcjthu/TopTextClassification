from model.model.LSTM import LSTM
from model.model.TextCNN import TextCNN
from model.model.SFKS.BasicCNN import BasicCNN
from model.model.SFKS.co_matching import CoMatching

model_list = {
    "LSTM": LSTM,
    "TextCNN": TextCNN,
    "SFKSCNN": BasicCNN,
    "Comatching": CoMatching
}


def get_model(name, config):
    if name in model_list.keys():
        return model_list[name](config)
    else:
        raise NotImplementedError
