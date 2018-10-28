from model.model.fnn import FNN

model_list = {
    "FNN": FNN
}


def get_model(name):
    if name in model_list.keys():
        pass
    else:
        raise NotImplementedError
