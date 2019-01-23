import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import json

from utils.util import calc_accuracy, generate_embedding


class BiDAF(nn.Module):
    def __init__(self, l, dim):
        super(BiDAF, self).__init__()

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
        self.article_encoder = GRUEncoder(self.batch * self.k * 4, self.word_size, self.hidden_size, self.layers,
                                          self.max_len)
        self.encoder = GRUEncoder(self.batch * self.k * 4, self.hidden_size * 8, self.hidden_size, self.layers,
                                  self.max_len)

        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))

        self.embs = nn.Embedding(self.word_num, self.word_size)
        if config.getboolean("data", "need_word2vec"):
            self.embs = generate_embedding(self.embs, config)

        self.relu = nn.ReLU()

        self.rank_module = nn.Linear(self.hidden_size * 8 * self.k * self.max_len, 1)

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

        y = self.rank_module(s).view(batch,-1)

        # print("y", y.size())

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
