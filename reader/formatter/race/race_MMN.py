import json
import torch
import numpy as np
import jieba
import random

from utils.util import check_multi


class RaceMMNFormatter:
    def __init__(self, config):
        self.need = config.getboolean("data", "need_word2vec")
        if self.need:
            self.word_dim = config.getint("data", "vec_size")
        else:
            self.word2id = json.load(open(config.get("data", "word2id"), "r"))

        self.max_len = config.getint('data', 'max_len')
        self.question_len = config.getint('data', 'question_max_len')
        self.option_len = config.getint('data', 'option_max_len')

        self.symbol = [",", ".", "?", "\""]
        self.last_symbol = [".", "?", "\""]

    def check(self, data, config):
        data = json.loads(data)
        return data

    def lookup(self, sent, max_len, transformer):
        lookup_id = []
        for word in sent:
            if not (word in self.word2id.keys()):
                if self.need:
                    lookup_id.append(transformer.load("UNK"))
                else:
                    lookup_id.append(self.word2id["UNK"])
            else:
                if self.need:
                    lookup_id.append(transformer.load(word))
                else:
                    lookup_id.append(self.word2id[word])
        while len(lookup_id) < max_len:
            if self.need:
                lookup_id.append(transformer.load('PAD'))
            else:
                lookup_id.append(self.word2id['PAD'])

        return lookup_id[:max_len]

    def parse(self, sent):
        result = []
        sent = sent.split(" ")
        for word in sent:
            for symbol in self.symbol:
                word = word.replace(symbol, "")
            if len(word) == 0:
                continue

            result.append(word)

        return result

    def format(self, data, config, transformer, mode):
        document = []
        option = []
        question = []
        label = []

        for temp_data in data:
            question.append(self.lookup(self.parse(temp_data["question"]), self.question_len, transformer))

            option.append([self.lookup(self.parse(temp_data["option"][0]), self.option_len, transformer),
                           self.lookup(self.parse(temp_data["option"][1]), self.option_len, transformer),
                           self.lookup(self.parse(temp_data["option"][2]), self.option_len, transformer),
                           self.lookup(self.parse(temp_data["option"][3]), self.option_len, transformer)])

            if temp_data["answer"] == "A":
                label_x = 0
            if temp_data["answer"] == "B":
                label_x = 1
            if temp_data["answer"] == "C":
                label_x = 2
            if temp_data["answer"] == "D":
                label_x = 3

            label.append(label_x)

            tmp_document = self.lookup(self.parse(temp_data["article"]), self.max_len, transformer)
            ttt = []
            for o in range(4):
                ttt.append([])
                ttt[-1].append(tmp_document)
            
            document.append(ttt)

        if self.need:
            question = torch.tensor(question, dtype = torch.float)
            document = torch.tensor(document, dtype = torch.float)
            option = torch.tensor(option, dtype = torch.float)
        else:
            question = torch.tensor(question, dtype = torch.long)
            document = torch.tensor(document, dtype = torch.long)
            option = torch.tensor(option, dtype = torch.long)
        label = torch.tensor(label, dtype = torch.long)
        

        return {'statement': question, 'label': label, 'reference': document, 'answer': option}

