import json
import torch
import numpy as np
import jieba
import random

from utils.util import check_multi


class RaceFormatter:
    def __init__(self, config):
        self.need = config.getboolean("data", "need_word2vec")
        self.word2id = json.load(open(config.get("data", "word2id"), "r"))
        self.max_len = config.getint("data", "max_len")

    def check(self, data, config):
        data = json.loads(data)
        if len(data["article"]) == 0:
            #print(len(data["article"]))
            # print("gg3")
            return None

        return data

    def lookup(self, data, transforemer=None):
        data = data.split(" ")
        lookup_id = []
        for word in data:
            word = word.replace("?", "").lower()
            if not (word in self.word2id.keys()):
                if self.need:
                    lookup_id.append(transforemer.load("UNK"))
                else:
                    lookup_id.append(self.word2id["UNK"])
            else:
                if self.need:
                    lookup_id.append(transforemer.load(word))
                else:
                    lookup_id.append(self.word2id[word])
        while len(lookup_id) < self.max_len:
            if self.need:
                lookup_id.append(transforemer.load("PAD"))
            else:
                lookup_id.append(self.word2id["PAD"])
        lookup_id = lookup_id[0:self.max_len]

        return lookup_id

    def format(self, data, config, transformer, mode):
        statement = []
        answer = []
        reference = []
        label = []

        for temp_data in data:
            statement.append(self.lookup(temp_data["question"], transformer))

            answer.append([self.lookup(temp_data["option"][0], transformer),
                           self.lookup(temp_data["option"][1], transformer),
                           self.lookup(temp_data["option"][2], transformer),
                           self.lookup(temp_data["option"][3], transformer)])

            if temp_data["answer"] == "A":
                label_x = 0
            if temp_data["answer"] == "B":
                label_x = 1
            if temp_data["answer"] == "C":
                label_x = 2
            if temp_data["answer"] == "D":
                label_x = 3

            label.append(label_x)

            temp_ref = []
            for option in ["A", "B", "C", "D"]:
                temp_ref.append([])
                for a in range(0, 1):
                    temp_ref[-1].append(self.lookup(temp_data["article"], transformer))

            reference.append(temp_ref)

        if self.need:
            statement = torch.tensor(statement, dtype=torch.float)
            reference = torch.tensor(reference, dtype=torch.float)
            answer = torch.tensor(answer, dtype=torch.float)
        else:
            statement = torch.tensor(statement, dtype=torch.long)
            reference = torch.tensor(reference, dtype=torch.long)
            answer = torch.tensor(answer, dtype=torch.long)

        label = torch.tensor(label, dtype=torch.long)

        return {"statement": statement, "label": label, "reference": reference, "answer": answer}
