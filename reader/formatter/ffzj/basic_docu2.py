import json
import torch
import numpy as np
import jieba
import random
import os
from pytorch_pretrained_bert import BertTokenizer


class FFZJBasicDocuFormatter2:
    def __init__(self, config):
        self.xd = {}
        xx = 30
        for a in range(0, 30):
            for b in range(a + 1, 30):
                xx += 1
                self.xd[(a, b)] = xx
        self.max_len = config.getint("data", "max_len")
        self.word2id = json.load(open(config.get("data", "word2id"), "r"))
        self.which = config.getint("data", "which")

    def check(self, data, config, mode):
        data = json.loads(data)
        if self.which in set(data["label"]):
            data["label"] = 1
        else:
            data["label"] = 0
        return data

    def transform(self, word):
        if not (word in self.word2id.keys()):
            return self.word2id["UNK"]
        else:
            return self.word2id[word]

    def convert(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.transform(token))

        while len(ids) < self.max_len:
            ids.append(self.transform("PAD"))
        return ids

    def load_file(self, name):
        res = []
        f = open(name, "r")
        for line in f:
            res = res + json.loads(line)
        return res

    def format(self, data, config, transformer, mode):
        input = []
        label = []
        for temp_data in data:
            res = self.load_file(os.path.join(config.get("data", "doc_path"), temp_data["name"] + ".txt"))
            res = res[0:self.max_len]

            indexed_tokens = self.convert(res)

            tokens_tensor = torch.tensor([indexed_tokens])

            input.append(tokens_tensor)

            labels = temp_data["label"]
            label.append(labels)

        input = torch.cat(input, dim=0)
        label = torch.LongTensor(np.array(label, dtype=np.int32))
        return {'input': input, 'label': label}
