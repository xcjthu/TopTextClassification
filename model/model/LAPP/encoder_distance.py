import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result, generate_embedding
from model.model.LAPP.Encoder.TextCNN import TextCNN
from model.model.LAPP.Encoder.LSTM import LSTM
from model.model.LAPP.Encoder.Bert import Bert


class Decoder(nn.Module):
    def __init__(self, config, size):
        super(Decoder, self).__init__()

        self.w = nn.Parameter(torch.Tensor(size, size))
        torch.nn.init.xavier_uniform_(self.w, gain=1)


    def forward(self, query, doc1, doc2):
        p1 = query.matmul(self.w)
        p1 = torch.bmm(p1.unsqueeze(1), doc1.unsqueeze(2)).squeeze(1)

        p2 = query.matmul(self.w)
        p2 = torch.bmm(p2.unsqueeze(1), doc2.unsqueeze(2)).squeeze(1)

        return torch.cat([p1, p2], dim = 1)


class Encoder_Distance(nn.Module):
    def __init__(self, config):
        super(Encoder_Distance, self).__init__()
        
        self.emb_dim = config.getint('data', 'vec_size')
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))
        self.embs = nn.Embedding(self.word_num, self.emb_dim)
        
        self.hidden = config.getint('model', 'hidden_size')

        if config.getboolean("data", "need_word2vec"):
            self.embs = generate_embedding(self.embs, config)
                    

        encoder = config.get('model', 'encoder')
        if encoder == 'textcnn':
            self.encoder = TextCNN(config)
        elif encoder == 'lstm':
            self.encoder = LSTM(config)
        elif encoder == 'bert':
            self.encoder = Bert(config)
        
        self.decoder = Decoder(config, self.hidden)
        
        
        self.encoder_name = encoder


    def init_multi_gpu(self, device):
        self.encoder = nn.DataParallel(self.encoder)
        self.decoder = nn.DataParallel(self.decoder)

    def forward(self, data, criterion, config, usegpu, acc_result = None):
        query = data['A']
        doc1 = data['B']
        doc2 = data['C']

        labels = data['label']
        
        if self.encoder_name != 'bert':
            query = self.embs(query)
            doc1 = self.embs(doc1)
            doc2 = self.embs(doc2)

        query = self.encoder(query)
        doc1 = self.encoder(doc1)
        doc2 = self.encoder(doc2)


        out_result = self.decoder(query, doc1, doc2)


        loss = criterion(out_result, labels)
        accu, acc_result = calc_accuracy(out_result, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(out_result, dim=1)[1].cpu().numpy(), "x": out_result, "accuracy_result": acc_result}


