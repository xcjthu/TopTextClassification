from model.model.fnn import FNN
from model.model.LSTM import LSTM

model_list = {
    "FNN": FNN,
    "LSTM": LSTM
}


def get_model(name):
    if name in model_list.keys():
        pass
    else:
        raise NotImplementedError
