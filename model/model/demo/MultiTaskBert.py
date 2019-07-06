import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import json

from utils.util import calc_accuracy, gen_result, generate_embedding
# from model.model.demo.TextCNN import TextCNN
from pytorch_pretrained_bert import BertModel

task_name_num = {
    'law': 183,
    'charge': 202,
    'time': 12
}


def get_num_class(task_name):
    return task_name_num[task_name]


class MultiTaskBert(nn.Module):
    def __init__(self, config):
        super(MultiTaskBert, self).__init__()

        self.taskName = config.get('data', 'task_name').split(',')
        self.taskName = [v.strip() for v in self.taskName]

        # self.cnn = TextCNN(config)
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))

        self.out = [768, get_num_class(name)) for name in self.taskName]

        self.out = nn.ModuleList(self.out)
    
    def init_multi_gpu(self, device):
        self.bert = nn.DataParallel(self.bert, device_ids = device)
        self.out = nn.DataParallel(self.out, device_ids = device)


    def forward(self, data, criterion, config, usegpu, acc_result={'law': None, 'charge': None, 'time': None}):
        passage = data['docs']  # batch, len
        labels = {}
        labels['law'] = data['label_law']
        labels['charge'] = data['label_charge']
        labels['time'] = data['label_time']
        labels['attribute'] = data['label_attr']

        # passage = self.embs(passage)

        # passage = self.cnn(passage)
        _, passage = self.bert(passage, output_all_encoded_layers = False)
        passage = passage.view(passage.size()[0], -1)


        task_result = {}
        for i in range(len(self.taskName)):
            task_result[self.taskName[i]] = self.out[i](passage)
            # task_result.append(self.out[i](vec))

        loss = criterion(task_result, labels)

        accu = {}
        accu['law'], acc_result['law'] = calc_accuracy(task_result['law'], labels['law'], config, acc_result['law'])
        accu['time'], acc_result['time'] = calc_accuracy(task_result['time'], labels['time'], config,
                                                         acc_result['time'])
        accu['charge'], acc_result['charge'] = calc_accuracy(task_result['charge'], labels['charge'], config,
                                                             acc_result['charge'])

        result = {
            'law': torch.max(task_result['law'], dim=1)[1].cpu().numpy(),
            'charge': torch.max(task_result['charge'], dim=1)[1].cpu().numpy(),
            'time': torch.max(task_result['time'], dim=1)[1].cpu().numpy()
        }

        return {"loss": loss, "accuracy": accu, "result": result, "x": task_result, "accuracy_result": acc_result}
