import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result, generate_embedding


class InputLayer(nn.Module):
    def __init__(self, config):
        super(InputLayer, self).__init__()

        self.vecsize = config.getint('data', 'vec_size')
        #self.hidden_size = config.getint('model', 'hidden_size')
        self.hidden_size = self.vecsize // 2

        bidirectional = True
        self.gru = nn.GRU(self.vecsize, self.hidden_size, batch_first = True, bidirectional = bidirectional)
        
        self.wh = nn.Linear(self.vecsize, self.vecsize, bias = False)
        self.we = nn.Linear(self.vecsize, self.vecsize)

        self.init_para()


    def init_para(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


    def forward(self, in_seq):
        gru_out, gru_hidden = self.gru(in_seq)

        out = self.wh(gru_out) + self.we(in_seq)

        reset_gate = F.sigmoid(out)

        # reset_gate = reset_gate.expand(in_seq.shape[0], in_seq.shape[1], in_seq.shape[2])
        return in_seq.mul(reset_gate) + gru_out.mul(1 - reset_gate)


class SoftSel(nn.Module):
    def __init__(self, config):
        super(SoftSel, self).__init__()

        self.hidden_size = config.getint('data', 'vec_size')

        self.wg = nn.Parameter(Var(torch.Tensor(self.hidden_size, self.hidden_size)))
        torch.nn.init.xavier_uniform_(self.wg, gain=1)
        
    def forward(self, hi1, hi2):
        G = hi1.matmul(self.wg)
        G = torch.bmm(G, torch.transpose(hi2, 1, 2))
        #G = torch.bmm(hi1, self.wg.unsqueeze(0).expand(hi1.shape[0], self.hidden_size, self.hidden_size))
        #G = torch.bmm(G, torch.transpose(hi2, 1, 2))
        
        G = F.softmax(G, dim = 2)
        return torch.bmm(G, hi2)


class Match(nn.Module):
    def __init__(self, config):
        super(Match, self).__init__()
        
        self.vecsize = config.getint('data', 'vec_size')
        self.hidden_size = config.getint('data', 'vec_size')

        self.match = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.linear_hm = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.match_gru = nn.GRU(self.hidden_size, self.hidden_size // 2, batch_first = True, bidirectional = True)
        
        self.linear_att = nn.Linear(self.hidden_size, 1)


    def forward(self, hi1, hi2, hi3, hi4):
        m1 = torch.cat([hi1.mul(hi2), hi1 - hi2], dim = 2)
        m2 = torch.cat([hi3.mul(hi4), hi3 - hi4], dim = 2)

        m1 = F.relu(self.match(m1))
        m2 = F.relu(self.match(m2))

        hm = self.linear_hm(torch.cat([m1, m2], dim = 2))

        hm, match_hidden = self.match_gru(hm)
        
        hmax, _ = torch.max(hm, dim = 1)
        
        hatt_softmax = F.softmax(F.relu(self.linear_att(hm)), dim = 1).squeeze().unsqueeze(1)
        hatt = torch.bmm(hatt_softmax, hm).squeeze()

        hf = torch.cat([hmax, hatt], dim = 1)
        return hf


class EAMatch(nn.Module):
    def __init__(self, config):
        super(EAMatch, self).__init__()
        self.match = Match(config)
        self.softsel = SoftSel(config)

    def forward(self, hq, ha):
        hq_ = self.softsel(hq, ha)
        hqa = self.softsel(ha, hq_)
        hqa_ = self.softsel(hqa, ha)
        ha_ = self.softsel(ha, hqa)
        hf = self.match(hqa, hqa_, ha, ha_)

        return hf



class MultiMatchNetWithoutPassage(nn.Module):
    def __init__(self, config):
        super(MultiMatchNetWithoutPassage, self).__init__()

        self.hidden_size = config.getint('data', 'vec_size')
        
        self.emb_dim = config.getint('data', 'vec_size')
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))

        self.embs = nn.Embedding(self.word_num, self.emb_dim)
        if config.getboolean("data", "need_word2vec"):
            self.embs = generate_embedding(self.embs, config)
        

        self.topN = config.getint('data', 'topN')
        self.batch_size = config.getint('train', 'batch_size')

        self.input = InputLayer(config)
       
        
        self.eam1 = EAMatch(config)
        self.eam2 = EAMatch(config)        

        self.output = nn.Linear(4 * self.hidden_size, 1)
        

        self.multi = config.getboolean('data', 'multi_choice')
        if self.multi:
            self.multi_module = nn.Linear(4, 16)

    def init_multi_gpu(self, device):
        self.input = nn.DataParallel(self.input)
        self.embs = nn.DataParallel(self.embs)
        self.softsel = nn.DataParallel(self.softsel)
        self.match = nn.DataParallel(self.match)
        self.output = nn.DataParallel(self.output)
        if self.multi:
            self.multi_module = nn.DataParallel(self.multi_module)


    def forward(self, data, criterion, config, usegpu, acc_result = None):
        question = data['statement']
        option_list = data['answer']
        labels = data['label']

        question = self.embs(question)
        
        hq = self.input(question)
        hq = hq.unsqueeze(1).repeat(1, 4, 1, 1)
        hq = hq.view(self.batch_size * 4, hq.shape[2], hq.shape[3])

        out = []
        feature = []

        option = option_list.view(self.batch_size * 4, option_list.shape[2])#, option_list.shape[3])
        option = self.embs(option)


        ha = self.input(option)
            
        hf1 = self.eam1(ha, hq)
        hf2 = self.eam2(hq, ha)


        hf = torch.cat([hf1, hf2], dim = 1)
        # print(hf.shape)
        score = self.output(hf)
        score = score.view(self.batch_size, 4)

        if self.multi:
            score = self.multi_module(score)

        out_result = score
        
        loss = criterion(out_result, labels)
        accu, acc_result = calc_accuracy(out_result, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(out_result, dim=1)[1].cpu().numpy(), "x": out_result, "accuracy_result": acc_result}



