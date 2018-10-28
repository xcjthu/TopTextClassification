import torch
import torch.nn as nn
import torch.functional as F

from utils.util import calc_accuracy, print_info


class transpose(nn.Module):
    def __init__(self):
        super(transpose, self).__init__()

    def forward(self, x):
        return (torch.transpose(x[0], 0, 1), torch.transpose(x[1], 0, 1))


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("model", "hidden_size")

        self.lstm = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True,
                            num_layers=config.getint("model", "num_layers"))

        self.fc = nn.Linear(self.hidden_dim, config.getint("model", "output_dim"))
        self.transpose = transpose()

    def init_hidden(self, config, usegpu):
        if torch.cuda.is_available() and usegpu:
            self.hidden = (
                torch.autograd.Variable(
                    torch.zeros(config.getint("train", "batch_size"), config.getint("model", "num_layers"),
                                self.hidden_dim).cuda()),
                torch.autograd.Variable(
                    torch.zeros(config.getint("train", "batch_size"), config.getint("model", "num_layers"),
                                self.hidden_dim).cuda()))
        else:
            self.hidden = (
                torch.autograd.Variable(
                    torch.zeros(config.getint("train", "batch_size"), config.getint("model", "num_layers"),
                                self.hidden_dim)),
                torch.autograd.Variable(
                    torch.zeros(config.getint("train", "batch_size"), config.getint("model", "num_layers"),
                                self.hidden_dim)))

    def init_multi_gpu(self, device):
        self.lstm = nn.DataParallel(self.lstm)
        self.fc = nn.DataParallel(self.fc)
        self.transpose = nn.DataParallel(self.transpose)

    def forward(self, data, criterion, config, usegpu):
        x = data["input"]
        labels = data["label"]

        x = x.view(config.getint("train", "batch_size"), -1, self.data_size)
        self.init_hidden(config, usegpu)
        self.hidden = self.transpose(self.hidden)

        lstm_out, self.hidden = self.lstm(x, self.hidden)

        lstm_out = torch.max(lstm_out, dim=1)[0]

        y = self.fc(lstm_out)

        loss = criterion(y, labels)
        accu = calc_accuracy(y, labels, config)

        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy()}
