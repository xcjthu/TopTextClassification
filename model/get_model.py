from model.model.fnn import FNN
from model.model.LSTM import LSTM

model_list = {
    "FNN": FNN,
    "LSTM": LSTM
}


def get_model(name, config):
    if name in model_list.keys():
        return model_list[name](config)
    else:
        raise NotImplementedError
