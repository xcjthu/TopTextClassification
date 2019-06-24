import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

from utils.util import calc_accuracy, print_info


class YSBert(nn.Module):
    def __init__(self, config):
        super(YSBert, self).__init__()
        self.output_dim = 0
        with open(config.get("data", "label_file"), "r") as f:
            for line in f:
                self.output_dim += 1

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.fc = nn.Linear(768, self.output_dim * 2)

    def init_multi_gpu(self, device):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        x = data['input']
        labels = data['label']

        _, y = self.bert(x, output_all_encoded_layers=False)
        y = y.view(y.size()[0], -1)
        y = self.fc(y)

        y = y.view(y.size()[0], -1, 2)
        y_out = nn.Softmax(dim=2)(y)
        y_out = y_out[:, :, 1]

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y_out, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
