import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.vec_size = config.getint('data', 'vec_size')
        self.hidden_size = config.getint('model', 'hidden_size')

        self.encoder = nn.LSTM(self.vec_size, self.hidden_size, batch_first = True)

    def forward(self, in_seq):
        out, _ = self.encoder(in_seq)
        return out


class SelfAtt(nn.Module):
    def __init__(self, config):
        super(SelfAtt, self).__init__()
        
        self.hidden_size = config.getint('model', 'hidden_size')
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, in_seq):
        softmax = self.linear(in_seq).squeeze(2)
        softmax = torch.softmax(softmax, dim = 1)
        return torch.bmm(softmax.unsqueeze(1), in_seq).squeeze(1)



class Reader(nn.Module):
    def __init__(self, config):
        super(Reader, self).__init__()

        self.hidden_size = config.getint('model', 'hidden_size')
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, batch_first = True)
        self.output = nn.Linear(self.hidden_size, 1)

    def forward(self, doc, que_self_att):
        out = que_self_att.unsqueeze(1).repeat(1, doc.shape[1], 1)
        out = torch.cat([doc, out], dim = 2)

        out, _ = self.gru(out)
        out, _ = torch.max(out, dim = 1)
        out = self.output(out)

        return out


class OneOption(nn.Module):
    def __init__(self, config):
        super(OneOption, self).__init__()
        
        self.hidden_size = config.getint('model', 'hidden_size')

        self.encoder = Encoder(config)
        self.self_att = SelfAtt(config)
        
        self.selector_w = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))

        self.reader = Reader(config)
    
    def init_multi_gpu(self, device):
        self.encoder = nn.DataParallel(self.encoder)
        self.self_att = nn.DataParallel(self.self_att)
        # self.selector_w = nn.DataParallel(self.selector_w)
        self.reader = nn.DataParallel(self.reader)


    def forward(self, doces, que):
        # doc: batch * doc_num * len * vec_size
        # que: batch * len * vec_size

        que = self.encoder(que)

        question = self.self_att(que)
        
        doc_prob = []
        reader_out = []
        for doc_index in range(doces.shape[1]):
            # selector part
            doc = doces[:,doc_index]
            doc = self.encoder(doc)

            selector_out = doc.matmul(self.selector_w)
            selector_out = torch.bmm(selector_out, question.unsqueeze(2))
            
            selector_out, _ = torch.max(selector_out, dim = 1)
            doc_prob.append(selector_out)
            
            # reader part
            doc_ans = self.reader(doc, question)
            reader_out.append(doc_ans.unsqueeze(1))

        doc_prob = torch.cat(doc_prob, dim = 1)
        doc_prob = torch.softmax(doc_prob, dim = 1) # batch * doc_num

        reader_out = torch.cat(reader_out, dim = 1) # batch * doc_num * 1
        
        
        return doc_prob, reader_out


class DSQA(nn.Module):
    def __init__(self, config):
        super(DSQA, self).__init__()

        self.oneOpt = OneOption(config)

    def init_multi_gpu(self, device):
        # self.oneOpt.init_multi_gpu(device)
        self.oneOpt = nn.DataParallel(self.oneOpt)

    def forward(self, data, criterion, config, usegpu, acc_result = None):
        
        question = data['statement'] # batch * 4 * len * vec_size
        # option_list = data['answer']
        documents = data['reference'] # batch * 4 * doc_num * len * vec_size
        labels = data['label'] # batch

        option_prob = []
        option_output = []

        doc_choice_all = []
        for option_index in range(4):
            que = question[:,option_index]
            doces = documents[:,option_index]

            
            doc_prob, reader_out = self.oneOpt(doces, que)
            _, doc_choice = torch.max(doc_prob, dim = 1)

            doc_choice_all.append(doc_choice.unsqueeze(1))
            # doc_prob: batch * doc_num
            # reader_out: batch * doc_num * 1
            opt_prob = torch.bmm(doc_prob.unsqueeze(1), torch.sigmoid(reader_out)).squeeze(1)
            option_prob.append(opt_prob)
            option_output.append(torch.bmm(doc_prob.unsqueeze(1), reader_out).squeeze(1))

        option_prob = torch.cat(option_prob, dim = 1)
        option_output = torch.cat(option_output, dim = 1)

        doc_choice_all = torch.cat(doc_choice_all, dim = 1)
        print(doc_choice_all.shape)
        
        '''
        print(option_prob.shape)
        print(option_output.shape)
        '''

        loss = criterion(option_prob, option_output, labels)
        out_result = option_prob
        accu, acc_result = calc_accuracy(out_result, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(out_result, dim=1)[1].cpu().numpy(), "x": out_result, 
            "accuracy_result": acc_result, 
            "doc_choice": doc_choice_all.cpu().numpy()
        }



