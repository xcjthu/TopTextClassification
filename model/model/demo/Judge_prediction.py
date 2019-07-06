import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import json

from model.model.demo.TextCNN import TextCNN

from utils.util import calc_accuracy, gen_result, generate_embedding

task_name_num = {
    'law': 183,
    'charge': 202,
    'time': 12
}

def get_num_class(task_name):
    return task_name_num[task_name]
    


class Attribute(nn.Module):
    def __init__(self, config):
        super(Attribute, self).__init__()
        
        self.attrNum = config.getint('data', 'attrNum')
        self.w = nn.Parameter(torch.Tensor(config.getint('model', 'hidden_size'), config.getint('model', 'hidden_size')))
        self.u = nn.Parameter(torch.Tensor(config.getint('data', 'attrNum'), config.getint('model', 'hidden_size')))
        
        torch.nn.init.xavier_uniform_(self.w, gain=1)
        torch.nn.init.xavier_uniform_(self.u, gain=1)
        
        self.out = [nn.Linear(config.getint('model', 'hidden_size'), 2) for i in range(self.attrNum)]
        
        self.out = nn.ModuleList(self.out)

    def forward(self, passage):
        # passage: batch, len, hidden_size
        # label: batch, attrNum, 1

        content = passage.matmul(self.w)
        content = content.matmul(torch.transpose(self.u, 0, 1))
        content = torch.transpose(content, 1, 2) # batch, attrNum, len
        content = torch.softmax(content, dim = 2)

        vec = torch.bmm(content, passage) # batch, attrNum, hidden_size
        
        result = []
        for i in range(self.attrNum):
            # result.append(self.out[i](vec[:,i]).unsqueeze(1))
            result.append(self.out[i](vec[:,i]))
        # result = torch.cat(result, dim = 1)
        vec, _ = torch.max(vec, dim = 1)
        return vec, result


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()

        self.w = nn.Parameter(torch.Tensor(config.getint('model', 'hidden_size'), config.getint('model', 'hidden_size')))
        self.u = nn.Parameter(torch.Tensor(config.getint('data', 'taskNum'), config.getint('model', 'hidden_size')))
        
        torch.nn.init.xavier_uniform_(self.w, gain=1)
        torch.nn.init.xavier_uniform_(self.u, gain=1)

    def forward(self, passage):
        content = passage.matmul(self.w)
        content = content.matmul(torch.transpose(self.u, 0, 1))
        content = torch.transpose(content, 1, 2)
        content = torch.softmax(content, dim = 2)

        return torch.bmm(content, passage)



class JudgePrediction(nn.Module):
    def __init__(self, config):
        super(JudgePrediction, self).__init__()
        
        self.emb_dim = config.getint('data', 'vec_size')
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))
        
        self.embs = nn.Embedding(self.word_num, self.emb_dim)
        if config.getboolean("data", "need_word2vec"):
            self.embs = generate_embedding(self.embs, config)
        
        self.taskName = config.get('data', 'task_name').split(',')
        self.taskName = [v.strip() for v in self.taskName]

        # self.gru = nn.LSTM(config.getint('data', 'vec_size'), config.getint('model', 'hidden_size'), batch_first = True)
        self.attr_task = False
        if config.get('train', 'type_of_loss') == 'demo_multi_task_loss':
            self.attr = Attribute(config)
            self.attr_task = True
            self.attr_encoder = nn.LSTM(config.getint('data', 'vec_size'), config.getint('model', 'hidden_size'), batch_first = True)

        self.encoder_type = config.get('model', 'encoder')
        if self.encoder_type == 'cnn':
            self.encoder = TextCNN(config)
        elif self.encoder_type == 'lstm':
            self.encoder = nn.LSTM(config.getint('data', 'vec_size'), config.getint('model', 'hidden_size'), batch_first = True)
        elif self.encoder_type == 'gru':
            self.encoder = nn.GRU(config.getint('data', 'vec_size'), config.getint('model', 'hidden_size'), batch_first = True)

        
        if self.attr_task:
            self.predictor = [nn.LSTMCell(2 * config.getint('model', 'hidden_size'), config.getint('model', 'hidden_size'), bias = True) for name in self.taskName]
        else:
            self.predictor = [nn.LSTMCell(config.getint('model', 'hidden_size'), config.getint('model', 'hidden_size'), bias = True) for name in self.taskName]

        self.cell_fc = [nn.Linear(config.getint('model', 'hidden_size'), config.getint('model', 'hidden_size')) for name in self.taskName]
        self.hidden_fc = [nn.Linear(config.getint('model', 'hidden_size'), config.getint('model', 'hidden_size')) for name in self.taskName]
        self.out = [nn.Linear(config.getint('model', 'hidden_size'), get_num_class(name)) for name in self.taskName]
        
        self.predictor = nn.ModuleList(self.predictor)
        self.cell_fc = nn.ModuleList(self.cell_fc)
        self.hidden_fc = nn.ModuleList(self.hidden_fc)
        self.out = nn.ModuleList(self.out)

    def forward(self, data, criterion, config, usegpu, acc_result = {'law': None, 'charge': None, 'time': None}):
        passage = data['docs']  # batch, len
        labels = {}
        labels['law'] = data['label_law']
        labels['charge'] = data['label_charge']
        labels['time'] = data['label_time']
        labels['attribute'] = data['label_attr']

        passage = self.embs(passage)
        
        if self.encoder_type == 'cnn':
            task_vec = self.encoder(passage)
        else:
            task_vec, _ = self.encoder(passage)
            task_vec, _ = torch.max(task_vec, dim = 1)

        if self.attr_task:
            attr, attr_result = self.attr(self.attr_encoder(passage)[0])
            task_vec = torch.cat([task_vec, attr], dim = 1)
        # print(task_vec.shape)

        # task_vec, _ = self.predictor(task_vec)   # batch, taskNum, hidden_size
        
        task_result = {}
        h = torch.zeros(passage.shape[0], config.getint('model', 'hidden_size')).cuda()
        c = torch.zeros(passage.shape[0], config.getint('model', 'hidden_size')).cuda()
        for i in range(len(self.taskName)):
            h, c = self.predictor[i](task_vec, (h, c))
            h = self.hidden_fc[i](h)
            c = self.cell_fc[i](h)
            task_result[self.taskName[i]] = self.out[i](h)
            # task_result.append(self.out[i](vec))
        
        if config.get('train', 'type_of_loss') == 'demo_multi_task_loss':
            loss = criterion(attr_result, task_result, labels)
        else:
            loss = criterion(task_result, labels)

        accu = {}
        accu['law'], acc_result['law'] = calc_accuracy(task_result['law'], labels['law'], config, acc_result['law'])
        accu['time'], acc_result['time'] = calc_accuracy(task_result['time'], labels['time'], config, acc_result['time'])
        accu['charge'], acc_result['charge'] = calc_accuracy(task_result['charge'], labels['charge'], config, acc_result['charge'])

        result = {
            'law': torch.max(task_result['law'], dim = 1)[1].cpu().numpy(),
            'charge': torch.max(task_result['charge'], dim = 1)[1].cpu().numpy(),
            'time': torch.max(task_result['time'], dim = 1)[1].cpu().numpy()
        }
        
        return {"loss": loss, "accuracy": accu, "result": result, "x": task_result, "accuracy_result": acc_result}



