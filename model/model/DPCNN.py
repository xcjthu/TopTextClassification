import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import calc_accuracy, print_info


class DPCNN(nn.Module):
    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.output_dim = config.getint("model", "output_dim")
        self.filters = config.getint('model', 'filters')
        self.batch_size = config.getint('train', 'batch_size')

        self.conv1 = nn.Conv2d(1, self.filters, (3, self.data_size))
        self.conv2 = nn.Conv2d(self.filters, self.filters, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.filters, self.output_dim)

    def init_hidden(self, config, usegpu):
        pass

    def init_multi_gpu(self, device):
        self.conv1 = nn.DataParallel(self.conv1, device_ids=device)
        self.conv2 = nn.DataParallel(self.conv2, device_ids=device)
        self.pooling = nn.DataParallel(self.pooling, device_ids=device)
        self.padding_conv = nn.DataParallel(self.padding_conv, device_ids=device)
        self.padding_pool = nn.DataParallel(self.padding_pool, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)
        self.relu = nn.DataParallel(self.relu, device_ids=device)

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        x = data['input']
        labels = data['label']

        x = x.view(self.batch_size, 1, -1, self.data_size)
        x = self.conv1(x)
        x = self.padding_conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.padding_conv(x)
        x = self.relu(x)
        x = self.conv2(x)

        while x.size()[-2] > 2:
            x = self.padding_pool(x)
            pooling_x = self.pooling(x)

            x = self.padding_conv(pooling_x)
            x = F.relu(x)
            x = self.conv2(x)
            x = self.padding_conv(x)
            x = F.relu(x)
            x = self.conv2(x)

            x = x + pooling_x

        x = x.view(self.batch_size, self.filters)
        y = self.fc(x)

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
