import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import calc_accuracy, print_info


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.output_dim = config.getint("model", "output_dim")
        self.batch_size = config.getint('train', 'batch_size')

        self.min_gram = config.getint("model", "min_gram")
        self.max_gram = config.getint("model", "max_gram")
        self.convs = []
        for a in range(self.min_gram, self.max_gram + 1):
            self.convs.append(nn.Conv2d(1, config.getint('model', 'filters'), (a, self.data_size)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = (self.max_gram - self.min_gram + 1) * config.getint('model', 'filters')
        self.fc = nn.Linear(self.feature_len, self.output_dim)
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

        if config.get('train', 'type_of_loss') == 'multi_label_cross_entropy_loss':
            self.multi = True
        else:
            self.multi = False

    def init_hidden(self, config, usegpu):
        pass

    def init_multi_gpu(self, device):
        # self.convs = [nn.DataParallel(v) for v in self.convs]
        for conv in self.convs:
            conv = nn.DataParallel(conv)
        # self.convs = nn.ModuleList(self.convs)
        self.fc = nn.DataParallel(self.fc)
        self.relu = nn.DataParallel(self.relu)
        self.sigmoid = nn.DataParallel(self.sigmoid)

    # pass

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        x = data['input']
        labels = data['label']

        x = x.view(self.batch_size, 1, -1, self.data_size)
        # print(x, labels)

        conv_out = []
        gram = self.min_gram
        # self.attention = []
        for conv in self.convs:
            y = self.relu(conv(x))
            y = torch.max(y, dim=2)[0].view(self.batch_size, -1)

            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        y = self.fc(conv_out)
        if self.multi:
            y = self.sigmoid(y)

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
