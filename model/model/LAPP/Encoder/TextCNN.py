import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        
        self.min_gram = config.getint("model", "min_gram")
        self.max_gram = config.getint("model", "max_gram")

        self.hidden = config.getint('model', 'hidden_size')

        self.convs = []
        for a in range(self.min_gram, self.max_gram + 1):
            self.convs.append(nn.Conv2d(1, config.getint('model', 'filters'), (a, self.data_size)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = (self.max_gram - self.min_gram + 1) * config.getint('model', 'filters')
        self.fc = nn.Linear(self.feature_len, self.hidden)
        self.relu = nn.ReLU()


    def forward(self, data):
        x = data
        
        batch_size = data.shape[0]
        x = x.view(x.shape[0], 1, -1, self.data_size)


        conv_out = []
        gram = self.min_gram
        
        for conv in self.convs:
            y = self.relu(conv(x))
            y = torch.max(y, dim=2)[0].view(batch_size, -1)

            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        y = self.fc(conv_out)
        
        return y
