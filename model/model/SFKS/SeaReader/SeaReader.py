import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from utils.util import calc_accuracy, gen_result

class Gate(nn.Module):
    def __init__(self, config, input_size):
        super(Gate, self).__init__()
        # self.gate_w = nn.Parameter(Var(torch.Tensor(input_size, 1).cuda()))
        self.gate_w = nn.Linear(input_size, 1)
        self.input_size = input_size

    def forward(self, seq, gate_control):
        gate_control = gate_control.contiguous()
        gate = self.gate_w(gate_control.view(-1, gate_control.shape[-1]))
        gate = F.sigmoid(gate)

        gate = gate.view(seq.shape[0],-1, 1).expand(seq.shape[0], seq.shape[1], seq.shape[2])

        return seq.mul(gate)


class SeaReader(nn.Module):
    def __init__(self, config):
        super(SeaReader, self).__init__()

        self.vecsize = config.getint('data', 'vec_size')
        self.statement_len = config.getint('data', 'max_len')
        self.option_len = config.getint('data', 'max_len')
        self.topN = config.getint('data', 'topN')
        self.doc_len = config.getint('data', 'max_len')

        self.batchsize = config.getint('train', 'batch_size')
        self.hidden_size = config.getint('model', 'hidden_size')

        self.context_layer = nn.GRU(self.vecsize, self.hidden_size, batch_first=True)  # , bidirectional = True)
        self.statement_reason_layer = nn.GRU(self.hidden_size, self.hidden_size,
                                             batch_first=True)  # , bidirectional = True)
        self.doc_reason_layer = nn.GRU(self.hidden_size * 2, self.hidden_size,
                                       batch_first=True)  # , bidirectional = True)

        self.reason_gate = Gate(config, self.hidden_size)
        # self.doc_reason_gate = Gate(config, usegpu, self.hidden_size * 2)
        self.intergrate_gate = nn.Linear(2 * self.hidden_size, 1)

        self.output_layer = nn.Linear(4 * self.hidden_size, 1)


    def init_multi_gpu(self, device):
        self.context_layer = nn.DataParallel(self.context_layer)
        self.statement_reason_layer = nn.DataParallel(self.statement_reason_layer)
        self.doc_reason_layer = nn.DataParallel(self.doc_reason_layer)
        self.reason_gate = nn.DataParallel(self.reason_gate)
        self.intergrate_gate = nn.DataParallel(self.intergrate_gate)
        self.output_layer = nn.DataParallel(self.output_layer)
        

    def init_hidden(self, config, usegpu):
        if (usegpu):
            self.context_statement_hidden = Var(torch.Tensor(1, self.batchsize * 4, self.hidden_size).cuda())
            self.context_doc_hidden = Var(torch.Tensor(1, self.batchsize * self.topN * 4, self.hidden_size).cuda())
            self.doc_reason_hidden = Var(torch.Tensor(1, self.batchsize, self.hidden_size).cuda())
            self.statement_reason_hidden = Var(torch.Tensor(1, self.batchsize, self.hidden_size).cuda())
        
	
    def forward(self, data, criterion, config, usegpu, acc_result = None):
        # self.init_hidden(config, usegpu)

        question = data['statement']
        option_list = data['answer']
        documents = data['reference']
        labels = data['label']
        # documents shape : batchsize * topn * doclength * vecsize
        

        out_result = []
        for option_index in range(4):
            option = option_list[:,option_index].contiguous()
            docs = documents[:,option_index].contiguous()

            statement = torch.cat([question, option], dim = 1)

            statement, self.context_statement_hidden =  self.context_layer(statement)
            
            docs, self.context_doc_hidden = self.context_layer(docs.view(self.batchsize * self.topN, self.doc_len, self.vecsize))
            docs = docs.view(self.batchsize, self.topN, self.doc_len, self.hidden_size)
            

            docs_read_info = []
            read_sum = []
            for doc_index in range(self.topN):
                doc = docs[:, doc_index]
                match_mat = torch.bmm(statement, torch.transpose(doc, 1, 2))  # batch_size, statement_len, doc_len

                softmax_col = F.softmax(match_mat, dim = 2)
                read_sum.append(torch.bmm(softmax_col, doc).unsqueeze(1))
                
                softmax_row = F.softmax(match_mat, dim = 1)
                doc_read = torch.bmm(torch.transpose(softmax_row, 1, 2), statement)
                docs_read_info.append(doc_read.unsqueeze(1))
            docs_read_info = torch.cat(docs_read_info, dim = 1)
            docs_read_info = torch.cat([docs, docs_read_info], dim = 3)
            
            read_sum = torch.cat(read_sum, dim = 1) # batchsize, topN, len, hidden_size


            doc_doc_att = []
            doc_doc_tmp = docs_read_info.view(self.batchsize, self.topN * self.doc_len, self.hidden_size * 2)
            for doc_index in range(self.topN):
                doc = docs_read_info[:, doc_index]
                match_m = torch.bmm(doc, torch.transpose(doc_doc_tmp, 1, 2))
                softmax_col = F.softmax(match_m, dim = 2)
                doc_doc_att.append(torch.bmm(softmax_col, doc_doc_tmp).unsqueeze(1))
            

            doc_read_info = torch.cat(doc_doc_att, dim = 1) # batchsize, topN, doc_len, 2 * hidden_size
            
            reason_statement_result = []
            reason_doc_result = []
            for doc_index in range(self.topN):
                doc = doc_read_info[:, doc_index]
                statement = read_sum[:, doc_index]

                statement, self.statement_reason_hidden = self.statement_reason_layer(statement)
                doc, self.doc_reason_hidden = self.doc_reason_layer(doc)
                
                statement = torch.max(statement, dim = 1)[0]
                doc = torch.max(doc, dim = 1)[0]
                
                reason_statement_result.append(statement.unsqueeze(1))
                reason_doc_result.append(doc.unsqueeze(1))

            # batchsize, topN, hidden_size
            reason_statement_result = torch.cat(reason_statement_result, dim = 1)
            reason_doc_result = torch.cat(reason_doc_result, dim = 1)

            result = torch.cat([reason_statement_result, reason_doc_result], dim = 2).view(self.batchsize * self.topN, 2 * self.hidden_size)
            
            coef = self.intergrate_gate(result).unsqueeze(2)
            result = torch.bmm(coef, result.unsqueeze(1)).view(self.batchsize, self.topN, 2 * self.hidden_size)

            result_max = torch.max(result, dim = 1)[0]
            result_mean = torch.mean(result, dim = 1)
            
            result = torch.cat([result_max, result_mean], dim = 1)

            out = self.output_layer(result)
            out_result.append(out)

        out_result = torch.cat(out_result, dim = 1)
        out_result = F.softmax(out_result, dim = 1)
        

        loss = criterion(out_result, labels)
        accu, acc_result = calc_accuracy(out_result, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(out_result, dim=1)[1].cpu().numpy(), "x": out_result, "accuracy_result": acc_result}






