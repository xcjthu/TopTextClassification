from model.model.LSTM import LSTM
from model.model.TextCNN import TextCNN
from model.model.bert import Bert

model_list = {
    "LSTM": LSTM,
    "TextCNN": TextCNN,
    "Bert": Bert
}


def get_model(name, config):
    if name in model_list.keys():
        return model_list[name](config)
    else:
        raise NotImplementedError
