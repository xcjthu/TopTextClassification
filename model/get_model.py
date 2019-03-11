from model.model.LSTM import LSTM
from model.model.TextCNN import TextCNN
from model.model.demo.Judge_prediction import JudgePrediction

model_list = {
    "LSTM": LSTM,
    "TextCNN": TextCNN,
    "JudgePrediction": JudgePrediction
}


def get_model(name, config):
    if name in model_list.keys():
        return model_list[name](config)
    else:
        raise NotImplementedError
