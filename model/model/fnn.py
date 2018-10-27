import torch
import torch.nn as nn
import torch.functional as F

from utils.util import calc_accuracy, print_info


class FNN(nn.Module):
    def __init__(self, config):
        super(FNN, self).__init__()

        self.fc1 = nn.Linear(config.getint("model", "in_classes"), 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, config.getint("model", "out_classes"))

    def init_hidden(self, config, usegpu):
        pass

    def init_multi_gpu(self, device):
        self.fc1 = nn.DataParallel(self.fc1, device_ids=device)
        self.relu = nn.DataParallel(self.relu, device_ids=device)
        self.fc2 = nn.DataParallel(self.fc2, device_ids=device)

    def forward(self, data, criterion):
        x = data["input"]
        labels = data["label"]

        x = self.fc2(self.relu(self.fc1(x)))

        loss = criterion(x, labels)
        accu = calc_accuracy(x, labels)

        return {"loss": loss, "accuracy": accu, "result": torch.max(x, dim=1)[1].cpu().numpy()}
