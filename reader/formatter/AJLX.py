import json
import torch
import numpy as np
from cutter.thulac_cutter import Thulac


class AJLXPredictionFormatter:
    def __init__(self, config):
        self.map_list = {
            "刑事": 0,
            "民事": 1,
            "行政": 2,
            "执行": 3
        }
        self.max_len = config.getint("data", "max_len")
        self.thulac = Thulac(config.get("cutter", "thulac_model"), config.get("cutter", "thulac_dict"))

    def check(self, data, config):
        data = json.loads(data)
        return data

    def format(self, data, config, transformer, mode):
        ss = []
        text = self.thulac.cut(data["text"])
        for a in range(0, len(text)):
            if a == self.max_len:
                break
            ss.append(transformer.load(text[a][0]))
        while len(ss) < self.max_len:
            ss.append(transformer.load("BLANK"))

        ss = torch.Tensor(ss)
        label = torch.LongTensor(np.array(data["type"], dtype=np.int32))

        return {'input': ss, 'label': label}
