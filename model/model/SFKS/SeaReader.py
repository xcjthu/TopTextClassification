import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

from utils.util import calc_accuracy


class Gate(nn.Module):
    def __init__(self, config, usegpu, input_size):
        super(Gate, self).__init__()
        # self.gate_w = nn.Parameter(Var(torch.Tensor(input_size, 1).cuda()))
        self.gate_w = nn.Linear(input_size, 1)
        self.input_size = input_size

    def forward(self, seq, gate_control):
        gate = self.gate_w(seq.view(-1, seq.shape[-1]))
        gate = F.sigmoid(gate)
        gate = gate.view(seq.shape[0], -1, 1)

        return torch.bmm(gate, seq)


class SeaReader(nn.Module):
    def __init__(self, config, usegpu):
        super(SeaReader, self).__init__()

        self.vecsize = config.getint('data', 'vecsize')
        self.statement_len = config.getint('data', 'statement_len')
        self.option_len = config.getint('data', 'option_len')
        self.topN = config.getint('data', 'topN')
        self.doc_len = config.getint('data', 'doc_len')

        self.batch_size = config.getint('train', 'batch_size')
        self.hidden_size = config.getint('train', 'hidden_size')

        self.context_layer = nn.GRU(self.vecsize, self.hidden_size, batch_first=True, bidirectional=True)
        self.statement_reason_layer = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.doc_reason_layer = nn.GRU(self.hidden_size * 2, self.hidden_size, batch_first=True, bidirectional=True)

        self.reason_gate = Gate(config, usegpu, self.hidden_size)
        # self.doc_reason_gate = Gate(config, usegpu, self.hidden_size * 2)
        self.intergrate_gate = nn.Linear(2 * self.hidden_size, 1)

        self.output_layer = nn.Linear(4 * self.hidden_size, 4)

    def init_hidden(self, config, usegpu):
        '''
        if (usegpu):
            self.context_statement_hidden = Var(torch.Tensor())
            self.context_doc_hidden =
            self.doc_reason_hidden =
            self.statement_reason_hidden =
        '''
        pass

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        question = data['statement']
        option_list = data['option_list']
        documents = data['documents']
        labels = data["label"]
        # documents shape : batchsize * topn * doclength * vecsize

        # 将问题描述和选项拼接在一起
        quetmp = question.unsqueeze(1).expand((-1, 4, -1, -1))
        statement = torch.cat([quetmp, option_list], dim=2)
        # statement size: batchsize * 4 * length * vecsize

        # cal matching Matrix (有一定的可能性这个写法有点问题)
        # statmp: batch_size * 4 * topn, self.statement_len + self.option_len, self.hidden_size
        statmp = torch.cat([statement.unsqueeze(2) for i in range(self.topN)], dim=2).view(-1,
                                                                                           self.statement_len + self.option_len,
                                                                                           self.vecsize)
        statmp, self.context_layer_statement_hidden = self.context_layer(statmp, self.context_statement_hidden)

        # docstmp: batch_szie * 4 * topn, self.doc_len, self.hidden_size
        docstmp = torch.cat([documents.unsqueeze(2) for i in range(4)], dim=2).view(-1, self.doc_len, self.vecsize)
        docstmp, self.context_doc_hidden = self.context_layer(docstmp, self.context_doc_hidden)

        # match_m: batch_size * 4 * topn, self.statement_len + self.option_len, self.doc_len
        match_m = torch.bmm(statmp, torch.transpose(docstmp, 1,
                                                    2))  # .view(self.batch_size, 4, self.topN, self.statement_len + self.option_len, self.doc_len)

        # question_centric path
        softmax_col = F.softmax(match_m, dim=2)
        read_summary = torch.bmm(softmax_col, docstmp)
        # read_summary: batchsize * 4 * topn, statement_len + option_len, hidden_size

        # document_centric path
        softmax_row = torch.transpose(F.softmax(match_m, dim=1), 1, 2)
        doc_read = torch.bmm(softmax_row, statmp)
        # doc_read: batch_size * 4 * topn, doc_len, 2*hidden_size
        doc_read = torch.cat([docstmp, doc_read], dim=2)

        doc_read = doc_read.view(self.batch_size * 4, self.topN * self.doc_len, 2 * self.hidden_size)
        # doc_read = torch.cat([doc_read.unsqueeze(1) for i in range(self.topN)]).view(self.batch_size * 4 * self.topN * self.topN, self.doc_len, 2 * self.hidden_size)
        # batch_size * 4 * topn,
        doc_match = torch.bmm(doc_read, torch.transpose(doc_read, 1, 2))
        softmax = F.softmax(doc_match, dim=2)
        doc_read = torch.bmm(softmax, doc_read).view(-1, self.topN * self.doc_len, 2 * self.hidden_size)

        # get matching feature
        # match_feature = torch.max(match_m, )

        # 现在已经有了
        # question centric path 获得的 read_summary batchsize * 4 * topn, statement_len + option_len, hidden_size
        # document centric path 获得的 doc_read  batchsize * 4, topN * doc_len, 2 * hidden_size
        # match feature 这一次实现先把这个忽略看看

        statement = self.reason_gate.forward(read_summary, statmp)
        documents = self.reason_gate.forward(doc_read, docstmp)

        statement, self.statement_reason_hidden = self.statement_reason_layer(statement, self.statement_reason_hidden)
        documents, self.doc_reason_hidden = self.doc_reason_layer(documents, self.doc_reason_hidden)

        statement = torch.max(statement, dim=1)[0]
        documents = torch.max(documents, dim=1)[0]

        documents = torch.cat([statement, documents], dim=1)

        documents = torch.bmm(documents.unsqueeze(2), self.intergrate_gate(documents).unsqueeze(2)).squeeze(2)
        documents = documents.view(self.batch_size * 4, self.topN, 2 * self.hidden_size)

        documents_max = torch.max(documents, dim=1)[0]
        documents_mean = torch.mean(documents, dim=1)

        final_feature = torch.cat([documents_max, documents_mean], dim=1)

        out = self.output_layer(final_feature)
        out = out.view(self.batch_size, 4)

        loss = criterion(out, labels)
        accu, acc_result = calc_accuracy(out, labels, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(out, dim=1)[1].cpu().numpy(), "x": out,
                "accuracy_result": acc_result}
