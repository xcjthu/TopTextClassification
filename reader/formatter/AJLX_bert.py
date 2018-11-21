import json
import torch
import numpy as np
import jieba
from pytorch_pretrained_bert import BertTokenizer


class AJLXBertPredictionFormatter:
    def __init__(self, config):
        self.map_list = {
            "刑事": 0,
            "民事": 1,
            "行政": 2,
            "执行": 3
        }
        self.max_len = config.getint("data", "max_len")

        self.tokenizer = BertTokenizer.from_pretrained('/data/disk3/private/zhx/bert/chinese/vocab.txt')

    def check(self, data, config):
        data = json.loads(data)
        if len(data["text"]) == 0:
            return None
        return data

    def convert(self, tokens):
        ids = []
        for token in tokens:
            if token in self.tokenizer.vocab.keys():
                ids.append(self.tokenizer.vocab[token])
            else:
                ids.append(self.tokenizer.vocab["[UNK]"])
        return ids

    def format(self, data, config, transformer, mode):
        input = []
        label = []
        for temp_data in data:
            ss = []
            res = temp_data["text"]
            for a in range(0, len(res)):
                ss = ss + [res[a]]
            ss = ss[0:self.max_len]

            while len(ss) < self.max_len:
                ss = ss + ["[PAD]"]

            indexed_tokens = self.convert(ss)
            tokens_tensor = torch.tensor([indexed_tokens])

            input.append(tokens_tensor)
            label.append(self.map_list[temp_data["type"]])

        input = torch.cat(input, dim=0)
        label = torch.LongTensor(np.array(label, dtype=np.int32))
        return {'input': input, 'label': label}
