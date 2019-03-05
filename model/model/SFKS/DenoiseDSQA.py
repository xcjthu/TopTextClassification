import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import json
from utils.util import calc_accuracy, generate_embedding


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.vec_size = config.getint('data', 'vec_size')
        self.hidden_size = config.getint('model', 'hidden_size')

        self.gru = nn.GRU(self.vec_size, self.hidden_size, batch_first = True, bidirectional = True)
        self.att_weight = nn.Parameter(torch.Tensor(2 * self.hidden_size, 1))
        torch.nn.init.xavier_uniform(self.att_weight, gain=1)


    def forward(self,in_seq, attention = False):
        out, _ = self.gru(in_seq)
        if attention:
            weight = out.matmul(self.att_weight).squeeze(2)
            weight = torch.softmax(weight, dim = 1).unsqueeze(1)
            out = torch.bmm(weight, out).squeeze(1)

        return out


class Selector_word_level(nn.Module):
    def __init__(self, config):
        super(Selector_word_level, self).__init__()
            
        self.hidden_size = config.getint('model', 'hidden_size')
        self.topN = config.getint('data', 'topN')

        self.weight = nn.Parameter(torch.Tensor(2 * self.hidden_size, 2 * self.hidden_size))
        torch.nn.init.xavier_uniform(self.weight, gain=1)

    def forward(self, question, passage):
        # passage : batch * topN * 4, len, 2 * hidden_size
        # question: batch * topN * 4, 2 * hidden_size

        result = passage.matmul(self.weight)
        result = torch.bmm(result, question.unsqueeze(2)).squeeze(2)
        result, _ = torch.max(result, dim = 1)
        result = result.view(-1, 4, self.topN)

        return result # torch.softmax(result, dim = 2) # batch, 4, topN


class Reader(nn.Module):
    def __init__(self, config):
        super(Reader, self).__init__()
        
        self.hidden_size = config.getint('model', 'hidden_size')
        self.topN = config.getint('data', 'topN')


        #self.multi = config.getboolean("data", "multi_choice")
        #if self.multi:
        #self.linear = nn.Linear(4, 16)


        self.att = nn.Parameter(torch.Tensor(2 * self.hidden_size, 2 * self.hidden_size))
        self.out = nn.Parameter(torch.Tensor(2 * self.hidden_size, 2 * self.hidden_size))
        torch.nn.init.xavier_uniform(self.out, gain=1)
        torch.nn.init.xavier_uniform(self.att, gain=1)

    def forward(self, question, passages):
        # question: batch * 4 * topN, 2 * hidden_size
        # passages: batch * 4 * topN, len, 2 * hidden_size

        attention = passages.matmul(self.att)
        attention = torch.bmm(attention, question.unsqueeze(2)).squeeze(2) 
        attention = torch.softmax(attention, dim = 1)
        passage_vec = torch.bmm(attention.unsqueeze(1), passages).squeeze(1)
        
        out = passage_vec.matmul(self.out)
        out = torch.bmm(out.unsqueeze(1), question.unsqueeze(2)).squeeze()
        
        ans_prob = out.view(-1, 4, self.topN)
        #ans_prob = torch.transpose(1, 2)
        #if self.multi:
        #    ans_prob = self.multi_module(ans_prob)

        return ans_prob


class DenoiseDSQA(nn.Module):
    def __init__(self, config):
        super(DenoiseDSQA, self).__init__()
        
        self.topN = config.getint('data', 'topN')
        self.hidden_size = config.getint('model', 'hidden_size')

        self.emb_dim = config.getint('data', 'vec_size')
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))
        self.embs = nn.Embedding(self.word_num, self.emb_dim)
        if config.getboolean("data", "need_word2vec"):
            self.embs = generate_embedding(self.embs, config)

        self.multi = config.getboolean('data', 'multi_choice')
        if self.multi:
            self.multi_module = nn.Linear(4, 16)


        self.encoder = Encoder(config)
        self.selector = Selector_word_level(config)
        self.reader = Reader(config)


    def init_multi_gpu(self, device):
        self.encoder = nn.DataParallel(self.encoder)
        self.selector = nn.DataParallel(self.selector)
        self.reader = nn.DataParallel(self.reader)


    def forward(self, data, criterion, config, usegpu, acc_result = None):
        question = data['statement']
        option = data['answer']
        passages = data['reference']
        labels = data['label']
        
        batch_size = question.shape[0]

        question = question.unsqueeze(1).repeat(1, 4, 1)
        question = question.view(batch_size * 4, question.shape[2])
        option = option.view(batch_size * 4, option.shape[2])
        

        question = torch.cat([question, option], dim = 1)
        question = self.embs(question)

        question = self.encoder(question, True)
        question = question.unsqueeze(1).repeat(1, self.topN, 1).view(batch_size * 4 * self.topN, 2 * self.hidden_size)
        
        passages = passages.view(batch_size * 4 * self.topN, passages.shape[3])
        passages = self.embs(passages)
        passages = self.encoder(passages)
        
        passage_prob = self.selector(question, passages) # batch, 4, topN
        ans_prob = self.reader(question, passages)      # batch, 4, topN

        final_prob = passage_prob.mul(ans_prob)
        out_result = torch.sum(final_prob, dim = 2) # batch, 4
        
        if self.multi:
            out_result = self.multi_module(out_result)
        
        out_result = torch.softmax(out_result, dim = 1)

        # print('out_result size', out_result.shape)
        # print('passage_prob size', passage_prob.shape)

        loss = criterion(out_result, torch.softmax(passage_prob, dim = 2), labels)


        accu, acc_result = calc_accuracy(out_result, labels, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(out_result, dim=1)[1].cpu().numpy(), "x": out_result, "accuracy_result": acc_result}


