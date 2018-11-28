import torch
import torch.nn as nn
import torch.functional as F

from utils.util import calc_accuracy, print_info

class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.hidden_size = config.getint("model", "hidden_size")
        self.output_dim = config.getint("model", "output_dim")
        self.num_layers = config.getint('model', 'num_layers')
        self.batch_size = config.getint('train', 'batch_size')
        self.kernel_size = config.getint('model', 'kernel_size')
        self.content_dim = config.getint('model', 'content_dim')
        self.linear_hidden_size = config.getint("model", "linear_hidden_size")

        self.lstm = nn.LSTM(input_size=self.data_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bias=True, batch_first=False, bidirectional=True)

        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_size * 2 + self.data_size,
                      out_channels=self.content_dim,
                      kernel_size=self.kernel_size),
            nn.BatchNorm1d(self.content_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.content_dim,
                      out_channels=self.content_dim,
                      kernel_size=self.kernel_size),
            nn.BatchNorm1d(self.content_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1*self.content_dim, self.linear_hidden_size),
            nn.BatchNorm1d(self.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear_hidden_size, self.output_dim)
        )

    def init_hidden(self, config, usegpu):
        pass

    def init_multi_gpu(self, device):
        self.lstm = nn.DataParallel(self.lstm, device_ids=device)
        self.convs = nn.DataParallel(self.convs, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        x = data['input']
        labels = data['label']

        x = x.view(self.batch_size, -1, self.data_size)

        out = self.lstm(x.permute(1, 0, 2))[0].permute(1, 2, 0)
        xc = x.permute(0, 2, 1)
        out = torch.cat((out, xc), dim=1)

        conv_out0 = self.convs(out)
        index = conv_out0.topk(1, dim=2)[1].sort(dim=2)[0]
        conv_out = conv_out0.gather(2, index)
        final_out = conv_out.view(conv_out.size(0), -1)
        y = self.fc(final_out)

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}