import json
import torch
import numpy as np
import os


class YSBertFormatter:
    labelToId = {}
    idToLabel = {}

    def __init__(self, config):
        self.label = {}
        with open(config.get("data", "label_file"), "r") as f:
            for line in f:
                self.label[line[:-1]] = len(self.label)

        self.word2id = {}
        with open(os.path.join(config.get("model", "bert_path"), "vocab.txt"), "r") as f:
            for line in f:
                self.word2id[line[:-1]] = len(self.word2id)

        self.max_len = config.getint("data", "max_len")

    def check(self, data, config):
        return json.loads(data)

    def lookup(self, passage, max_len):
        lookup_id = []
        for word in passage:
            try:
                lookup_id.append(self.word2id[word])
            except:
                lookup_id.append(self.word2id["[UNK]"])

        while len(lookup_id) < max_len:
            lookup_id.append(self.word2id["[PAD]"])
        lookup_id = lookup_id[:max_len]

        return lookup_id

    def format(self, data, config, transformer, mode):
        input = []
        label = []
        for x in data:
            input.append(self.lookup(x["sentence"], self.max_len))

            se = set(x["labels"])
            temp = []
            for k in self.label:
                if k in se:
                    temp.append(1)
                else:
                    temp.append(0)

            label.append(temp)

        input = torch.LongTensor(input)
        label = torch.LongTensor(label)

        return {'input': input, 'label': label}
