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
            torch.autograd.Variable(torch.zeros(2 * config.getint("model", "num_layers"), bs, self.hidden_dim)).cuda(),
            torch.autograd.Variable(torch.zeros(2 * config.getint("model", "num_layers"), bs, self.hidden_dim)).cuda())

    def init_multi_gpu(self, device):
        self.lstm = nn.DataParallel(self.lstm, device_ids=device)
        self.fc = nn.DataParallel(self.fc, device_ids=device)
        self.transpose = nn.DataParallel(self.transpose, device_ids=device)

    def forward(self, x, config):
        bs = x.size()[0]
        x = x.view(bs, -1, self.data_size)
        self.init_hidden(config, bs)

        lstm_out, self.hidden = self.lstm(x, self.hidden)

        return lstm_out


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()

        hz = config.getint("model", "hidden_size")

        self.w = nn.Linear(hz * 2, hz * 2)

    def forward(self, hq, hp):
        gq = torch.bmm(self.w(hq), torch.transpose(hp, 1, 2))
        gq = torch.softmax(gq, dim=2)

        bar_hq = torch.bmm(torch.transpose(hq, 1, 2), gq)
        bar_hq = torch.transpose(bar_hq, 1, 2)

        return bar_hq


class Comatch(nn.Module):
    def __init__(self, config):
        super(Comatch, self).__init__()

        hz = config.getint("model", "hidden_size")
        self.attq = Attention(config)
        self.atta = Attention(config)

        self.wm = nn.Linear(4 * hz, 2 * hz)

        self.relu = nn.ReLU()

    def subdim(self, a, b):
        return torch.cat([a - b, a * b], dim=2)

    def forward(self, hq, hp, ha):
        bar_hq = self.attq(hq, hp)
        bar_ha = self.atta(ha, hp)

        mq = self.relu(self.wm(self.subdim(bar_hq, hp)))
        ma = self.relu(self.wm(self.subdim(bar_ha, hp)))

        c = torch.cat([mq, ma], dim=2)
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
        self.lstm_c = BiLSTMEncoder(config)
        self.lstm_c.data_size = 4 * self.hidden_size
        self.lstm_c.lstm = nn.LSTM(self.lstm_c.data_size, self.lstm_c.hidden_dim, batch_first=True,
                                   num_layers=config.getint("model", "num_layers"), bidirectional=True)

        self.predictor = nn.Linear(2 * self.hidden_size, 1)

        self.co_match = Comatch(config)

        if config.get('train', 'type_of_loss') == 'multi_label_cross_entropy_loss':
            self.multi = True
        else:
            self.multi = False

    def init_multi_gpu(self, device):
        pass

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        q = data["statement"]
        a = data["answer"]
        p = data["reference"]
        labels = data["label"]

        bs = q.size()[0]
        p = p[:, 0]

        if not (config.getboolean("data", "need_word2vec")):
            q = self.embedding(q)
            a = self.embedding(a)
            p = self.embedding(p)


        a = a.view(bs * 4, -1, self.word_size)
        p = p.view(bs * 4, -1, self.word_size)

        hp = self.lstm_p(p, config)
        hq = self.lstm_q(q, config)
        ha = self.lstm_a(a, config)

        hp = hp.contiguous().view(bs, 4, -1, self.hidden_size * 2)
        ha = ha.contiguous().view(bs, 4, -1, self.hidden_size * 2)

        y_list = []

        for a in range(0, 4):
            p_temp = hp[:, a, :, :].view(bs, -1, self.hidden_size * 2)
            a_temp = ha[:, a, :, :].view(bs, -1, self.hidden_size * 2)
            c = self.co_match(hq, p_temp, a_temp)
            H = self.lstm_c(c, config)
            h = torch.max(H, dim=1)[0].view(bs, -1)
            y_list.append(self.predictor(h))

        y = torch.cat(y_list, dim=1)

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}


class CoMatching2(nn.Module):
    def __init__(self, config):
        super(CoMatching2, self).__init__()

        self.word_size = config.getint("data", "vec_size")
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))
        self.output_dim = config.getint("model", "output_dim")
        self.hidden_size = config.getint("model", "hidden_size")
        self.bs = config.getint("train", "batch_size")

        self.embedding = nn.Embedding(self.word_num, self.word_size)

        self.lstm_p = BiLSTMEncoder(config)
        self.lstm_a = BiLSTMEncoder(config)
        self.lstm_q = BiLSTMEncoder(config)
        self.lstm_c = BiLSTMEncoder(config)
        self.lstm_c.data_size = 4 * self.hidden_size
        self.lstm_c.lstm = nn.LSTM(self.lstm_c.data_size, self.lstm_c.hidden_dim, batch_first=True,
                                   num_layers=config.getint("model", "num_layers"), bidirectional=True)

        self.predictor = nn.Linear(2 * 10 * self.hidden_size, 1)

        self.co_match = Comatch(config)

        if config.get('train', 'type_of_loss') == 'multi_label_cross_entropy_loss':
            self.multi = True
        else:
            self.multi = False

    def init_multi_gpu(self, device):
        pass

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        q = data["statement"]
        a = data["answer"]
        p = data["reference"]
        labels = data["label"]

        bs = q.size()[0]

        if not (config.getboolean("data", "need_word2vec")):
            q = self.embedding(q)
            a = self.embedding(a)
            p = self.embedding(p)

        a = a.view(bs * 4, -1, self.word_size)
        p = p.view(bs * 4 * 10, -1, self.word_size)

        hp = self.lstm_p(p, config)
        hq = self.lstm_q(q, config)
        ha = self.lstm_a(a, config)

        hp = hp.contiguous().view(bs, 4, 10, -1, self.hidden_size * 2)
        ha = ha.contiguous().view(bs, 4, -1, self.hidden_size * 2)

        y_list = []

        for a in range(0, 4):
            a_temp = ha[:, a, :, :].view(bs, -1, self.hidden_size * 2)
            h_list = []
            for b in range(0, 10):
                p_temp = hp[:, a, b, :, :].view(bs, -1, self.hidden_size * 2)
                c = self.co_match(hq, p_temp, a_temp)
                H = self.lstm_c(c, config)
                h = torch.max(H, dim=1)[0].view(bs, -1)
                h_list.append(h)

            h = torch.cat(h_list, dim=1)
            y_list.append(self.predictor(h))

        y = torch.cat(y_list, dim=1)

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
