import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result, generate_embedding
from model.model.LAPP.Encoder.Bert import Bert


class DataMining(nn.Module):
    def __init__(self, config):
        super(DataMining, self).__init__()
        
        self.emb_dim = config.getint('data', 'vec_size')
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))
        self.embs = nn.Embedding(self.word_num, self.emb_dim)
        
        self.hidden = config.getint('model', 'hidden_size')

        if config.getboolean("data", "need_word2vec"):
            self.embs = generate_embedding(self.embs, config)
                    

        self.encoder = Bert(config)
        
        self.decoder = nn.Linear(self.hidden, 3)


    def init_multi_gpu(self, device):
        self.encoder = nn.DataParallel(self.encoder)
        self.decoder = nn.DataParallel(self.decoder)


    def forward(self, data, criterion, config, usegpu, acc_result = None):
        docs = data['ab']
        labels = data['label']
        docs = self.encoder(docs)

        out_result = self.decoder(docs)


        loss = criterion(out_result, labels)
        accu, acc_result = calc_accuracy(out_result, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(out_result, dim=1)[1].cpu().numpy(), "x": out_result, "accuracy_result": acc_result}


