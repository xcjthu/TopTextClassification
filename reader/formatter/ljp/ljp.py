import json
import torch
import numpy as np
import os

from utils.util import print_info


class LJPBertFormatter:
    labelToId = {}
    idToLabel = {}

    def __init__(self, config):
        min_freq = config.getint("data", "min_freq")
        self.crit_label = {}
        with open(config.get("data", "crit_label"), "r") as f:
            for line in f:
                arr = line[:-1].split(" ")
                label = arr[0].replace("[", "").replace("]", "")
                cnt = int(arr[1])
                if cnt >= min_freq:
                    self.crit_label[label] = len(label)

        self.law_label = {}
        with open(config.get("data", "law_label"), "r") as f:
            for line in f:
                arr = line[:-1].split(" ")
                x1 = int(arr[0])
                x2 = int(arr[1])
                cnt = int(arr[2])
                label = (x1, x2)
                if cnt >= min_freq:
                    self.law_label[label] = len(label)
        print_info("%d %d" % (len(self.crit_label), len(self.law_label)))

        self.word2id = {}
        with open(os.path.join(config.get("model", "bert_path"), "vocab.txt"), "r", encoding="utf8") as f:
            for line in f:
                self.word2id[line[:-1]] = len(self.word2id)

        self.task = config.get("data", "task")

        self.max_len = config.getint("data", "max_len")

    def check_crit(self, data):
        cnt = 0
        for x in data:
            if x in self.crit_label.keys():
                cnt += 1
            else:
                return False
        return cnt == 1

    def check_law(self, data):
        arr = []
        for x, y, z in data:
            if x < 102 or x > 452:
                continue
            if not ((x, y) in self.law_label.keys()):
                return False
            arr.append((x, y))

        arr = list(set(arr))
        arr.sort()

        cnt = 0
        for x in arr:
            if x in arr:
                cnt += 1  # return False
        return cnt == 1

    def check(self, data, config):
        data = json.loads(data)
        if len(data["meta"]["criminals"]) != 1:
            return None
        if len(data["meta"]["crit"]) == 0 or len(data["meta"]["law"]) == 0:
            return None
        if not (self.check_crit(data["meta"]["crit"])):
            return None
        if not (self.check_law(data["meta"]["law"])):
            return None

        return data

    def get_crit_id(self, data):
        for x in data:
            if x in self.crit_label.keys():
                return self.crit_label[x]

    def get_law_id(self, data):
        for x in data:
            y = (x[0], x[1])
            if y in self.law_label.keys():
                return self.law_label[y]

    def get_time_id(self, data):
        v = 0
        if len(data["youqi"]) > 0:
            v1 = data["youqi"][-1]
        else:
            v1 = 0
        if len(data["guanzhi"]) > 0:
            v2 = data["guanzhi"][-1]
        else:
            v2 = 0
        if len(data["juyi"]) > 0:
            v3 = data["juyi"][-1]
        else:
            v3 = 0
        v = max(v1, v2, v3)

        if data["sixing"]:
            opt = 0
        elif data["wuqi"]:
            opt = 0
        elif v > 10 * 12:
            opt = 1
        elif v > 7 * 12:
            opt = 2
        elif v > 5 * 12:
            opt = 3
        elif v > 3 * 12:
            opt = 4
        elif v > 2 * 12:
            opt = 5
        elif v > 1 * 12:
            opt = 6
        elif v > 9:
            opt = 7
        elif v > 6:
            opt = 8
        elif v > 0:
            opt = 9
        else:
            opt = 10

        return opt

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
            text = ""
            for y in x["content"]:
                for z in y:
                    text += z
            input.append(self.lookup(z, self.max_len))

            if self.task == "crit":
                l = self.get_crit_id(x)
            elif self.task == "law":
                l = self.get_law_id(x)
            else:
                l = self.get_law_id(x)

            label.append(l)

        input = torch.LongTensor(input)
        label = torch.LongTensor(label)

        return {'input': input, 'label': label}
