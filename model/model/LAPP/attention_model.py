import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var

import json
from utils.util import calc_accuracy, gen_result, generate_embedding


class Attention_model(nn.Module):
    def __init__(self, config):
        super(Attention_model, self).__init__()

        self.emb_dim = config.getint('data', 'vec_size')
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))
        self.embs = nn.Embedding(self.word_num, self.emb_dim)

        if config.getboolean("data", "need_word2vec"):
            self.embs = generate_embedding(self.embs, config)
    
        self.hidden = config.getint('model', 'hidden_size')
        self.encoder = nn.LSTM(self.emb_dim, self.hidden, batch_first = True)
        
        self.multi_head = nn.MultiheadAttention(self.hidden, config.getint('model', 'head_num'))

        self.w = nn.Parameter(torch.Tensor(self.hidden, self.hidden))
        torch.nn.init.xavier_uniform_(self.w, gain=1)


    def forward(self, data, criterion, config, usegpu, acc_result = None):
        query = data['A']
        doc1 = data['B']
        doc2 = data['C']
        labels = data['label']

        query = self.embs(query)
        doc1 = self.embs(doc1)
        doc2 = self.embs(doc2)

        query, _ = self.encoder(query)
        doc1, _ = self.encoder(doc1)
        doc2, _ = self.encoder(doc2)
        
        doc1_out, _ = self.multi_head(query.transpose(0, 1), doc1.transpose(0, 1), doc1.transpose(0, 1))
        doc2_out, _ = self.multi_head(query.transpose(0, 1), doc2.transpose(0, 1), doc2.transpose(0, 1))
        
        doc1_out = doc1_out.transpose(0, 1)
        doc2_out = doc2_out.transpose(0, 1)

        doc1_out = torch.max(doc1_out, dim = 1)[0]
        doc2_out = torch.max(doc2_out, dim = 1)[0]
        query = torch.max(query, dim = 1)[0]

        
        p1 = query.matmul(self.w)
        p1 = torch.bmm(p1.unsqueeze(1), doc1_out.unsqueeze(2)).squeeze(1)

        p2 = query.matmul(self.w)
        p2 = torch.bmm(p2.unsqueeze(1), doc2_out.unsqueeze(2)).squeeze(1)


        out_result = torch.cat([p1, p2], dim = 1)

        loss = criterion(out_result, labels)
        accu, acc_result = calc_accuracy(out_result, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(out_result, dim=1)[1].cpu().numpy(), "x": out_result, "accuracy_result": acc_result}

