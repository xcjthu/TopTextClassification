import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from utils.util import calc_accuracy, gen_result


class transpose(nn.Module):
    def __init__(self):
        super(transpose, self).__init__()

    def forward(self, x):
        return (torch.transpose(x[0], 0, 1), torch.transpose(x[1], 0, 1))


class BiLSTMEncoder(nn.Module):
    def __init__(self, config):
        super(BiLSTMEncoder, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("model", "hidden_size")

        self.lstm = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True,
                            num_layers=config.getint("model", "num_layers"), bidirectional=True)

        self.fc = nn.Linear(self.hidden_dim, config.getint("model", "output_dim"))
        self.transpose = transpose()

    def init_hidden(self, config, bs):
        self.hidden = (
            torch.autograd.Variable(torch.zeros(bs, 2 * config.getint("model", "num_layers"), self.hidden_dim)),
            torch.autograd.Variable(torch.zeros(bs, 2 * config.getint("model", "num_layers"), self.hidden_dim)))

    def init_multi_gpu(self, device):
        self.lstm = nn.DataParallel(self.lstm, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)
        self.transpose = nn.DataParallel(self.transpose, device_ids=device)

    def forward(self, x, config):
        bs = x.size()[0]
        x = x.view(bs, -1, self.data_size)
        self.init_hidden(config, bs)
        self.hidden = self.transpose(self.hidden)
        x = x.contiguous()
        self.hidden = self.hidden.contiguous()

        lstm_out, self.hidden = self.lstm(x, self.hidden)

        return lstm_out, self.hidden


class Comatch(nn.Module):
    def __init__(self, config):
        super(Comatch, self).__init__()

        self.wg = nn.Linear(1, 1)
        self.wm = nn.Linear(1, 1)
        self.relu = nn.ReLU()

    def subdim(self, a, b):
        return torch.cat([a - b, a * b], dim=1)

    def forward(self, hq, hp, ha):
        gq = torch.softmax(torch.bmm(self.wg(hq), hp))
        ga = torch.softmax(torch.bmm(self.wg(ha), ha))

        bar_hq = torch.bmm(hq, gq)
        bar_ha = torch.bmm(ha, ga)

        mq = self.relu(self.subdim(bar_hq, hp))
        ma = self.relu(self.subdim(bar_ha, hp))

        c = torch.cat([mq, mq], dim=1)
        return c


class CoMatching(nn.Module):
    def __init__(self, config):
        super(CoMatching, self).__init__()

        self.word_size = config.getint("data", "vec_size")
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))
        self.output_dim = config.getint("model", "output_dim")
        self.hidden_size = config.getint("model", "hidden_size")
        self.bs = config.getint("train", "batch_size")

        self.embedding = nn.Embedding(self.word_num, self.word_size)

        self.lstm_p = BiLSTMEncoder(config)
        self.lstm_a = BiLSTMEncoder(config)
        self.lstm_q = BiLSTMEncoder(config)

    def init_multi_gpu(self, device):
        pass

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        q = data["statement"]
        a = data["answer"]
        p = data["reference"]

        bs = q.size()[0]

        q = self.embedding(q)
        a = self.embedding(a)
        p = self.embedding(p)

        a = a.view(bs * 4, -1, self.word_size)
        p = p.view(bs * 4, -1, self.word_size)

        hp = self.lstm_p(p, config)
        hq = self.lstm_q(q, config)
        ha = self.lstm_a(a, config)
        print(hp.size())
        print(hq.size())
        print(ha.size())
        gg

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
