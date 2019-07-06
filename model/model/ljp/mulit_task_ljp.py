import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

from utils.util import calc_accuracy, print_info


class MultiTaskLJPBert(nn.Module):
    def __init__(self, config):
        super(MultiTaskLJPBert, self).__init__()
        min_freq = config.getint("data", "min_freq")
        self.crit_label = {}
        with open(config.get("data", "crit_label"), "r") as f:
            for line in f:
                arr = line[:-1].split(" ")
                label = arr[0].replace("[", "").replace("]", "")
                cnt = int(arr[1])
                if cnt >= min_freq:
                    self.crit_label[label] = len(label)

        self.law_label = {}
        with open(config.get("data", "law_label"), "r") as f:
            for line in f:
                arr = line[:-1].split(" ")
                x1 = int(arr[0])
                x2 = int(arr[1])
                cnt = int(arr[2])
                label = (x1, x2)
                if cnt >= min_freq:
                    self.law_label[label] = len(label)


        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.crit_fc = nn.Linear(768, len(self.crit_label)
        self.law_fc = nn.Linear(768, len(self.law_label))
        self.time_fc = nn.Linear(768, 11)

        self.out = [self.crit_fc, self.law_fc, self.time_fc]
        for i in range(10):
            self.out.append(nn.Linear(768, 2))
        self.out = nn.ModuleList(self.attr)

    def init_multi_gpu(self, device):
        self.bert = nn.DataParallel(self.bert, device_ids=device)
        #self.crit_fc = nn.DataParallel(self.crit_fc, device_ids = device)
        #self.law_fc = nn.DataParallel(self.law_fc, device_ids = device)
        #self.time_fc = nn.DataParallel(self.time_fc, device_ids = device)
        self.out = nn.DataParallel(self.out, device_ids = device)
    

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        x = data['input']
        labels = data['label']

        _, y = self.bert(x, output_all_encoded_layers=False)
        y = y.view(y.size()[0], -1)
        
        task_result = []
        for i in range(13):
            task_result.append(self.out[i](y))
        

        

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": y, "x": y,
                "accuracy_result": acc_result}
