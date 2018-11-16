import json
import torch
import numpy as np


class AJLXPredictionFormatter:
    def __init__(self, config):
        self.map_list = {
            "刑事": 0,
            "民事": 1,
            "行政": 2,
            "执行": 3
        }
        self.max_len = config.getint("data", "max_len")

    def check(self, data, config):
        data = json.loads(data)
        return data

    def format(self, data, config, transformer, mode):
        ss = []
        for a in range(0, len(data["text"])):
            if a == self.max_len:
                break
            ss.append(transformer.load(data["text"][a]))
        while len(ss) < self.max_len:
            ss.append(transformer.load("BLANK"))

        ss = torch.Tensor(ss)
        label = torch.LongTensor(np.array(data["type"], dtype=np.int32))

        return {'input': ss, 'label': label}
