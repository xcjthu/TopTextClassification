import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import json

from utils.util import calc_accuracy, gen_result, generate_embedding


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.batch_size = config.getint('train', 'batch_size')

        self.min_gram = config.getint("model", "min_gram")
        self.max_gram = config.getint("model", "max_gram")
        self.convs = []
        for a in range(self.min_gram, self.max_gram + 1):
            self.convs.append(nn.Conv2d(1, config.getint('model', 'filters'), (a, self.data_size)))

        self.convs = nn.ModuleList(self.convs)
        self.relu = nn.ReLU()
        self.feature_len = (self.max_gram - self.min_gram + 1) * config.getint('model', 'filters')
        self.fc = nn.Linear(self.feature_len, config.getint('model', 'hidden_size'))


    def forward(self, x):

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
        
        return self.fc(conv_out)






