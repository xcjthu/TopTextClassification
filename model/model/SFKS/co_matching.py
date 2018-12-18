'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.
Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.
This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import json
from utils.util import calc_accuracy


def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i, :, :seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)

    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result


class MatchNet(nn.Module):
    def __init__(self, mem_dim, dropoutP):
        super(MatchNet, self).__init__()
        self.map_linear = nn.Linear(2 * mem_dim, 2 * mem_dim)
        self.trans_linear = nn.Linear(mem_dim, mem_dim)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs):
        proj_p, proj_q, seq_len = inputs
        trans_q = self.trans_linear(proj_q)
        att_weights = proj_p.bmm(torch.transpose(proj_q, 1, 2))
        att_norm = masked_softmax(att_weights, seq_len)

        att_vec = att_norm.bmm(proj_q)
        elem_min = att_vec - proj_p
        elem_mul = att_vec * proj_p
        all_con = torch.cat([elem_min, elem_mul], 2)
        output = nn.ReLU()(self.map_linear(all_con))
        return output


class MaskLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP=0.3):
        super(MaskLSTM, self).__init__()
        self.lstm_module = nn.LSTM(in_dim, out_dim, layers, batch_first=batch_first, bidirectional=bidirectional,
                                   dropout=dropoutP)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs):
        input, seq_lens = inputs
        mask_in = input.new(input.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask_in[i, :seq_lens[i]] = 1
        mask_in = Variable(mask_in, requires_grad=False)

        input_drop = self.drop_module(input * mask_in)

        H, _ = self.lstm_module(input_drop)

        mask = H.new(H.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask[i, :seq_lens[i]] = 1
        mask = Variable(mask, requires_grad=False)

        output = H * mask

        return output


class CoMatch(nn.Module):
    def __init__(self, config):
        super(CoMatch, self).__init__()
        self.emb_dim = config.getint("data", "vec_size")  # 300
        self.mem_dim = config.getint("model", "hidden_size")  # 150
        self.dropoutP = config.getfloat("model", "dropout")  # args.dropoutP 0.2
        # self.cuda_bool = args.cuda

        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))

        self.embs = nn.Embedding(self.word_num, self.emb_dim)
        # self.embs.weight.data.copy_(corpus.dictionary.embs)
        # self.embs.weight.requires_grad = False

        self.encoder = MaskLSTM(self.emb_dim, self.mem_dim, dropoutP=self.dropoutP)
        self.l_encoder = MaskLSTM(self.mem_dim * 8, self.mem_dim, dropoutP=self.dropoutP)
        self.h_encoder = MaskLSTM(self.mem_dim * 2, self.mem_dim, dropoutP=0)

        self.match_module = MatchNet(self.mem_dim * 2, self.dropoutP)
        self.rank_module = nn.Linear(self.mem_dim * 2, 1)

        self.drop_module = nn.Dropout(self.dropoutP)

        self.more = config.getboolean("model", "one_more_softmax")

    def forward(self, inputs):
        documents, questions, options = inputs
        d_word, d_h_len, d_l_len = documents
        o_word, o_h_len, o_l_len = options
        q_word, q_len = questions

        # if self.cuda_bool: d_word, d_h_len, d_l_len, o_word, o_h_len, o_l_len, q_word, q_len = d_word.cuda(), d_h_len.cuda(), d_l_len.cuda(), o_word.cuda(), o_h_len.cuda(), o_l_len.cuda(), q_word.cuda(), q_len.cuda()
        # d_embs = self.drop_module(Variable(self.embs(d_word), requires_grad=False))
        # o_embs = self.drop_module(Variable(self.embs(o_word), requires_grad=False))
        # q_embs = self.drop_module(Variable(self.embs(q_word), requires_grad=False))

        d_embs = self.drop_module(self.embs(d_word))
        o_embs = self.drop_module(self.embs(o_word))
        q_embs = self.drop_module(self.embs(q_word))

        d_hidden = self.encoder(
            [d_embs.view(d_embs.size(0) * d_embs.size(1), d_embs.size(2), self.emb_dim), d_l_len.view(-1)])
        o_hidden = self.encoder(
            [o_embs.view(o_embs.size(0) * o_embs.size(1), o_embs.size(2), self.emb_dim), o_l_len.view(-1)])
        q_hidden = self.encoder([q_embs, q_len])

        d_hidden_3d = d_hidden.view(d_embs.size(0), d_embs.size(1) * d_embs.size(2), d_hidden.size(-1))
        d_hidden_3d_repeat = d_hidden_3d.repeat(1, o_embs.size(1), 1).view(d_hidden_3d.size(0) * o_embs.size(1),
                                                                           d_hidden_3d.size(1), d_hidden_3d.size(2))

        do_match = self.match_module([d_hidden_3d_repeat, o_hidden, o_l_len.view(-1)])
        dq_match = self.match_module([d_hidden_3d, q_hidden, q_len])

        dq_match_repeat = dq_match.repeat(1, o_embs.size(1), 1).view(dq_match.size(0) * o_embs.size(1),
                                                                     dq_match.size(1), dq_match.size(2))

        co_match = torch.cat([do_match, dq_match_repeat], -1)

        co_match_hier = co_match.view(d_embs.size(0) * o_embs.size(1) * d_embs.size(1), d_embs.size(2), -1)

        l_hidden = self.l_encoder([co_match_hier, d_l_len.repeat(1, o_embs.size(1)).view(-1)])
        l_hidden_pool, _ = l_hidden.max(1)

        h_hidden = self.h_encoder([l_hidden_pool.view(d_embs.size(0) * o_embs.size(1), d_embs.size(1), -1),
                                   d_h_len.view(-1, 1).repeat(1, o_embs.size(1)).view(-1)])
        h_hidden_pool, _ = h_hidden.max(1)

        o_rep = h_hidden_pool.view(d_embs.size(0), o_embs.size(1), -1)
        output = self.rank_module(o_rep).squeeze(2)
        if self.more:
            output = torch.nn.functional.log_softmax(output)

        return output


class CoMatching(nn.Module):
    def __init__(self, config):
        super(CoMatching, self).__init__()

        self.co_match = CoMatch(config)

    def init_multi_gpu(self, device):
        pass

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        q, ql = data["question"], data["question_len"]
        o, oh, ol = data["option"], data["option_sent"], data["option_len"]
        d, dh, dl = data["document"], data["document_sent"], data["document_len"]
        label = data["label"]

        x = [[d, dh, dl], [q, ql], [o, oh, ol]]
        y = self.co_match(x)

        loss = criterion(y, label)
        accu, acc_result = calc_accuracy(y, label, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}


"""
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

        self.predictor = nn.Linear(8 * self.hidden_size, 4)

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

        a = a.contiguous().view(bs * 4, -1, self.word_size)
        p = p.contiguous().view(bs * 4, -1, self.word_size)
        q = q.contiguous()

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
            y_list.append(h)

        y = torch.cat(y_list, dim=1)
        y = self.predictor(y)

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

        a = a.contiguous().view(bs * 4, -1, self.word_size)
        p = p.contiguous().view(bs * 4 * 10, -1, self.word_size)
        q = q.contiguous()

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
"""
