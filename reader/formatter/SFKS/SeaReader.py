import json
import torch
import numpy as np
import jieba

from utils.util import check_multi


class SeaReaderFormatter:
    def __init__(self, config):
        self.need = config.getboolean("data", "need_word2vec")
        self.word2id = json.load(open(config.get("data", "word2id"), "r"))
        self.max_len = config.getint("data", "max_len")

    def check(self, data, config):
        data = json.loads(data)
        if not ("answer" in data.keys()):
            # print("gg1")
            return None
        if len(data["answer"]) != 1:
            # print("gg2")
            return None
        if len(data["statement"]) == 0 or len(data["statement"]) > self.max_len:
            # print("gg3")
            return None
        for option in data["option_list"]:
            if len(data["option_list"][option]) == 0 or len(data["option_list"][option]) > self.max_len:
                # print("gg4")
                return None
        if len(data["option_list"]) != 4:
            # print("gg5")
            return None
        if not ("reference" in data.keys()):
            return None

        return data

    def lookup(self, data, transforemer=None):
        lookup_id = []
        for word in data:
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
            statement.append(self.lookup(temp_data["statement"], transformer))
            answer.append([self.lookup(temp_data["option_list"]["A"], transformer),
                           self.lookup(temp_data["option_list"]["B"], transformer),
                           self.lookup(temp_data["option_list"]["C"], transformer),
                           self.lookup(temp_data["option_list"]["D"], transformer)])

            if check_multi(config):
                label_x = [0, 0, 0, 0]
                if "A" in temp_data["answer"]:
                    label_x[0] = 1
                if "B" in temp_data["answer"]:
                    label_x[1] = 1
                if "C" in temp_data["answer"]:
                    label_x[2] = 1
                if "D" in temp_data["answer"]:
                    label_x[3] = 1
            else:
                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x = 0
                if "B" in temp_data["answer"]:
                    label_x = 1
                if "C" in temp_data["answer"]:
                    label_x = 2
                if "D" in temp_data["answer"]:
                    label_x = 3

            label.append(label_x)

            temp_ref = []
            for option in ["A", "B", "C", "D"]:
                temp_ref.append([])
                for a in range(0, 10):
                    temp_ref[-1].append(self.lookup(temp_data["reference"][option][a], transformer))

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
