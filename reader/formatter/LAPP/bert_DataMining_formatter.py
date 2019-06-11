import json
import random
import torch
import os

class DataMining_Bert_Formatter:
    def __init__(self, config):
        self.need = config.getboolean('data', 'need_word2vec')
        
        self.word2id = {}
        f = open(os.path.join(config.get("model", "bert_path"), "vocab.txt"), "r")
        
        for line in f:
            self.word2id[line.strip()] = len(self.word2id)

        self.max_len = config.getint('data', 'max_len')

    def check(self, data, config):
        data = json.loads(data)
        return data


    def lookup(self, data, max_len):
        look_id = []
        if len(data) > max_len:
            data = data[- max_len:]

        for word in data:
            if not word in self.word2id.keys():
                look_id.append(self.word2id['[UNK]'])
            else:
                look_id.append(self.word2id[word])
        
        while len(look_id) < max_len:
            look_id.append(self.word2id['[PAD]'])
        look_id = look_id[:max_len]

        return look_id
    
    def lookupab(self, a, b, max_len):
        look_id = [self.word2id['[CLS]']]
        look_id += self.lookup(a, max_len)
        look_id.append(self.word2id['[SEP]'])
        look_id += self.lookup(b, max_len)
        look_id.append(self.word2id['[SEP]'])
        
        return look_id
        

    def format(self, batch_data, config, transformer, mode):
        
        label = []
        ab = []

        l2id = {'unrelated': 0, 'agreed': 1, 'disagreed': 2}
        for data in batch_data:
            ab.append(self.lookupab(data['a'], data['b'], self.max_len))
            # A.append(self.lookup(''.join(data['a']), self.max_len))
            # B.append(self.lookup(''.join(data['b']), self.max_len))
            # C.append(self.lookup(''.join(data['C']), self.max_len))
            label.append(l2id[data['label']])

        # A = torch.tensor(A, dtype = torch.long)
        # B = torch.tensor(B, dtype = torch.long)
        # C = torch.tensor(C, dtype = torch.long)
        
        ab = torch.tensor(ab, dtype = torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return {'label': label, 'ab': ab}

