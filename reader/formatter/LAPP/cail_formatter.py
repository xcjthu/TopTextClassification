import json
import random
import torch

class LAPP_Formatter:
    def __init__(self, config):
        self.need = config.getboolean('data', 'need_word2vec')
        self.word2id = json.load(open(config.get('data', 'word2id'), 'r'))

        self.max_len = config.getint('data', 'max_len')

    def check(self, data, config):
        data = json.loads(data)
        return data


    def lookup(self, data, max_len):
        look_id = []

        for word in data:
            if not word in self.word2id.keys():
                look_id.append(self.word2id['UNK'])
            else:
                look_id.append(self.word2id[word])
        
        while len(look_id) < max_len:
            look_id.append(self.word2id['PAD'])
        look_id = look_id[:max_len]

        return look_id

    def format(self, batch_data, config, transformer, mode):
        A = []
        B = []
        C = []
        label = []
        
        l2id = {'B': 0, 'C': 1}
        for data in batch_data:
            A.append(self.lookup(data['A'], self.max_len))
            B.append(self.lookup(data['B'], self.max_len))
            C.append(self.lookup(data['C'], self.max_len))
            label.append(l2id[data['label']])

        A = torch.tensor(A, dtype = torch.long)
        B = torch.tensor(B, dtype = torch.long)
        C = torch.tensor(C, dtype = torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return {'label': label, 'A': A, 'B': B, 'C': C}

