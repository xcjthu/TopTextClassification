import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertForPreTraining

from utils.util import calc_accuracy, print_info

task_name_num = {
    'law': 183,
    'charge': 202,
    'time': 12
}


class BertDemo(nn.Module):
    def __init__(self, config):
        super(BertDemo, self).__init__()
        self.batch_size = config.getint('train', 'batch_size')

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.bert.half()

        self.taskName = config.get('data', 'task_name').split(',')
        self.taskName = [v.strip() for v in self.taskName]

        self.fc_list = []
        for a in range(0, len(self.taskName)):
            self.fc_list.append(nn.Linear(768, task_name_num[self.taskName[a]]))

        self.fc_list = nn.ModuleList(self.fc_list)
        self.fc_list.half()

    def init_multi_gpu(self, device):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, criterion, config, usegpu, acc_result={'law': None, 'charge': None, 'time': None}):
        x = data['docs']  # batch, len
        labels = {}
        labels['law'] = data['label_law']
        labels['charge'] = data['label_charge']
        labels['time'] = data['label_time']
        labels['attribute'] = data['label_attr']

        _, y = self.bert(x, output_all_encoded_layers=False)
        y = y.view(y.size()[0], -1)
        task_result = {}
        for a in range(0, len(self.taskName)):
            task_result[self.taskName[a]] = self.fc_list[a](y)

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
