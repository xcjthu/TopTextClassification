import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import json

from utils.util import calc_accuracy, generate_embedding


class BiDAF2(nn.Module):
    def __init__(self, l, dim):
        super(BiDAF2, self).__init__()

        self.word_size = dim

        self.w1 = nn.Linear(self.word_size, l)
        self.w2 = nn.Linear(self.word_size, l)

    def forward(self, h, q):
        max_len = h.size()[1]
        w1h = self.w1(h)
        w2q = self.w2(q)
        w3hq = torch.bmm(h, q.permute(0, 2, 1))

        # w1h = w1h.repeat(1, 1, max_len)
        w2q = w2q.permute(0, 2, 1)

        # print("w1h", w1h.size())
        # print("w2q", w2q.size())
        # print("w3hq", w3hq.size())

        a = w1h + w2q + w3hq

        # print("a", a.size())

        p = torch.softmax(a, dim=2)

        # print("p", p.size())
        c = torch.bmm(p, q)
        # print("c", c.size())

        m = torch.max(a, dim=2)[0]
        # print("m", m.size())
        p = torch.softmax(m, dim=1)
        # print("p", p.size())

        qc = h * p.view(p.size()[0], p.size()[1], 1).repeat(1, 1, h.size()[2])
        # print("qc", qc.size())

        return torch.cat([h, c, h * c, qc * c], dim=2)


class BiDAF(nn.Module):
    def __init__(self, l, dim):
        super(BiDAF, self).__init__()

        self.word_size = dim

        self.att_weight_c = nn.Linear(self.word_size, 1)
        self.att_weight_q = nn.Linear(self.word_size, 1)
        self.att_weight_cq = nn.Linear(self.word_size, 1)

    def forward(self, c, q):
        """
                    :param c: (batch, c_len, hidden_size * 2)
                    :param q: (batch, q_len, hidden_size * 2)
                    :return: (batch, c_len, q_len)
                    """
        c_len = c.size(1)
        q_len = q.size(1)

        # (batch, c_len, q_len, hidden_size * 2)
        # c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
        # (batch, c_len, q_len, hidden_size * 2)
        # q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
        # (batch, c_len, q_len, hidden_size * 2)
        # cq_tiled = c_tiled * q_tiled
        # cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

        cq = []
        for i in range(q_len):
            # (batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            # (batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)

        # (batch, c_len, q_len)
        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq

        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)
        # (batch, 1, c_len)
        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        q2c_att = torch.bmm(b, c).squeeze()
        # (batch, c_len, hidden_size * 2) (tiled)
        q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

        # (batch, c_len, hidden_size * 8)
        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
        return x


class GRUEncoder(nn.Module):
    def __init__(self, batch, input_size, output_size, layers, max_len):
        super(GRUEncoder, self).__init__()

        self.batch = batch
        self.input_size = input_size
        self.output_size = output_size
        self.len = max_len
        self.layers = layers

        self.encoder = nn.GRU(input_size, output_size, layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        hidden = torch.autograd.Variable(torch.zeros(self.layers * 2, self.batch, self.output_size).cuda())

        o, h = self.encoder(x, hidden)

        return o


class SimpleAndEffective(nn.Module):
    def __init__(self, config):
        super(SimpleAndEffective, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_size = config.getint("data", "vec_size")
        self.k = config.getint("data", "topk")
        self.batch = config.getint("train", "batch_size")
        self.layers = config.getint("model", "num_layers")
        self.max_len = config.getint("data", "max_len")

        self.bi_attention = BiDAF(self.max_len, 2 * self.hidden_size)
        self.bi_attention2 = BiDAF(self.max_len, 2 * self.hidden_size)

        self.question_encoder = GRUEncoder(self.batch * self.k * 4, self.word_size, self.hidden_size, self.layers,
                                           self.max_len)
        self.article_encoder = self.question_encoder  # GRUEncoder(self.batch * self.k * 4, self.word_size, self.hidden_size, self.layers,
        # self.max_len)
        self.encoder = GRUEncoder(self.batch * self.k * 4, self.hidden_size * 8, self.hidden_size, self.layers,
                                  self.max_len)

        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))

        self.embs = nn.Embedding(self.word_num, self.word_size)
        if config.getboolean("data", "need_word2vec"):
            self.embs = generate_embedding(self.embs, config)

        self.relu = nn.ReLU()

        if config.get("model", "rank_method") == "all":
            self.rank_module = nn.Linear(self.hidden_size * 8 * self.k * self.max_len, 1)
        else:
            self.rank_module = nn.Linear(self.hidden_size * 8 * self.max_len, 1)

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        question = data["question"]
        article = data["article"]
        labels = data["label"]

        batch = self.batch
        option = question.size()[1]
        k = self.k

        question = question.repeat(1, 1, k).view(batch * option * k, self.max_len)
        article = article.view(batch * option * k, self.max_len)

        # print("question", question.size())
        # print("article", article.size())

        question = self.embs(question)
        article = self.embs(article)

        # print("question", question.size())
        # print("article", article.size())

        question = self.question_encoder(question)
        article = self.article_encoder(article)

        # print("question", question.size())
        # print("article", article.size())

        attention = self.bi_attention(question, article)

        # print("attention", attention.size())

        attention = self.relu(attention)

        attention_x = self.encoder(attention)
        attention_x = self.bi_attention2(attention_x, attention_x)
        attention_x = self.relu(attention_x)

        # print("attention_x", attention_x.size())

        s = attention + attention_x

        # print("s", s.size())

        s = s.view(batch, option, -1)

        # print("s", s.size())

        if config.get("model", "rank_method") == "all":
            y = self.rank_module(s).view(batch, -1)
        elif config.get("model", "rank_method") == "max":
            y = s.view(batch * option, k, -1)
            y = torch.max(y, dim=1)[0]
            y = y.view(batch * option, -1)
            y = self.rank_module(y)
            y = y.view(batch, option)
        else:
            y = s.view(batch * option * k, -1)
            y = self.rank_module(y)
            y = y.view(batch, option, k)
            y = torch.max(y, dim=2)[0]
            y = y.view(batch, option)

        # print("y", y.size())

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
