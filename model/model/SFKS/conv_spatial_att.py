import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result



class InputLayer(nn.Module):
    def __init__(self, config):
        super(InputLayer, self).__init__()
        
        self.vec_size = config.getint('data', 'vec_size')
        self.hidden_size = config.getint('model', 'hidden_size')

        self.gru = nn.GRU(self.vec_size, self.hidden_size)

    def forward(self, in_seq):
        out, _ = self.gru(in_seq)
        return out



class EnrichAttention(nn.Module):
    def __init__(self, config, len1, len2):
        super(EnrichAttention, self).__init__()
        
        self.len1 = len1
        self.len2 = len2
        self.att_len = config.getint('model', 'att_size')
        self.hidden_size = config.getint('model', 'hidden_size')

        self.w1 = nn.Linear(self.hidden_size, self.att_len, bias = False) # nn.Parameter(torch.Tensor(self.att_len, self.hidden_size))
        self.w2 = nn.Linear(self.hidden_size, self.att_len, bias = False)#nn.Parameter(torch.)
        
        self.D = nn.Parameter(torch.Tensor(self.att_len, self.att_len))
        self.W = nn.Parameter(torch.Tensor(self.len1, self.len2))
        
        self.gru = nn.GRU(2 * self.hidden_size, self.hidden_size, batch_first = True)

    def forward(self, x1, x2):
        # first step
        w1 = F.relu(self.w1(x1))
        w2 = F.relu(self.w2(x2))
        w1 = w1.matmul(self.D)
        #w1 = torch.bmm(w1, self.D.unsqueeze(0).expand(x1.shape[0], self.att_len, self.att_len))
        
        M = torch.bmm(w1, torch.transpose(w2, 1, 2))
        M = M.mul(self.W.unsqueeze(0).expand(x1.shape[0], self.len1, self.len2))
        
        M = F.softmax(M, dim = 1)
        M = torch.bmm(M, x2)

        M = torch.cat([x1, M], dim = 2)

        M, hidden = self.gru(M)
        return M



class ConvSpatialAtt(nn.Module):
    def __init__(self, config):
        super(ConvSpatialAtt, self).__init__()
        self.input = InputLayer(config)

        self.que_len = config.getint('data', 'question_max_len')
        self.doc_len = config.getint('data', 'max_len')
        self.opt_len = config.getint('data', 'option_max_len')


        print('que_len:', self.que_len)
        print('doc_len:', self.doc_len)
        print('opt_len:', self.opt_len)

        self.emb_dim = config.getint('data', 'vec_size')
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))
        self.embs = nn.Embedding(self.word_num, self.emb_dim)
        

        self.kernel0 = 3
        self.kernel1 = 4
        self.kernel2 = 5
        

        self.enrichCQ = EnrichAttention(config, self.opt_len, self.que_len)
        self.enrichCP = EnrichAttention(config, self.opt_len, self.doc_len)
        self.enrichQP = EnrichAttention(config, self.que_len, self.doc_len)
        self.enrichQ = EnrichAttention(config, self.que_len, self.que_len)
        
        
        self.conv0 = nn.Conv2d(6, 6, (self.kernel0, self.que_len))
        self.conv1 = nn.Conv2d(6, 6, (self.kernel1, self.que_len))
        self.conv2 = nn.Conv2d(6, 6, (self.kernel2, self.que_len))

        self.linear_att = nn.Linear(3 * self.opt_len + 3 - self.kernel0 - self.kernel1 - self.kernel2, 1)
        self.linear_out = nn.Linear(3 * self.opt_len + 3 - self.kernel0 - self.kernel1 - self.kernel2, 1)


    def forward(self, data, criterion, config, usegpu, acc_result = None):
        question = data['statement']
        option_list = data['answer']
        documents = data['reference']
        labels = data['label']

        if not config.getboolean("data", "need_word2vec"):
            question = self.embs(question)

        question = self.input(question)
        out_result = []
        for option_index in range(4):
            option = option_list[:,option_index].contiguous()
            
            if not config.getboolean('data', 'need_word2vec'):
                option = self.embs(option)

            option = self.input(option)

            docs = documents[:,option_index].contiguous()

            out_all_doc = []
            doc_softmax = []
            #for doc_index in range(docs.shape[1]):
            for doc_index in range(1):

                doc = docs[:,doc_index].contiguous()
                
                if not config.getboolean('data', 'need_word2vec'):
                    doc = self.embs(doc)

                doc = self.input(doc)

                rcq = self.enrichCQ(option, question)
                rcp = self.enrichCP(option, doc)
                rqp = self.enrichQP(question, doc)
                rq = self.enrichQ(question, rqp)

                rq = torch.transpose(rq, 1, 2)
                rqp = torch.transpose(rqp, 1, 2)

                m11 = torch.bmm(rcq, rq)
                m12 = torch.bmm(rcp, rq)
                m13 = torch.bmm(option, rq)

                m21 = torch.bmm(rcq, rqp)
                m22 = torch.bmm(rcp, rqp)
                m23 = torch.bmm(option, rqp)

                
                M = torch.cat([m11.unsqueeze(1), m12.unsqueeze(1), m13.unsqueeze(1), m21.unsqueeze(1), m22.unsqueeze(1), m23.unsqueeze(1)], dim = 1)
                
                o1 = self.conv0(M)
                o1, _ = torch.max(o1, dim = 1) 

                o2 = self.conv1(M)
                o2, _ = torch.max(o2, dim = 1)

                o3 = self.conv2(M)
                o3, _ = torch.max(o3, dim = 1)
                
                '''
                print('o1.shape', o1.shape)
                print('o2.shape', o2.shape)
                print('o3.shape', o3.shape)
                '''
                
                feature = torch.cat([o1, o2, o3], dim = 1).squeeze()
                doc_softmax.append(self.linear_att(feature))
                out_all_doc.append(feature.unsqueeze(1))
            
            doc_softmax = torch.cat(doc_softmax, dim = 1)
            doc_softmax = F.softmax(doc_softmax, dim = 1).unsqueeze(1)

            out_all_doc = torch.cat(out_all_doc, dim = 1)
            
            #print('doc_softmax', doc_softmax.shape)
            #print('out_all_doc', out_all_doc.shape)

            feature = torch.bmm(doc_softmax, out_all_doc).squeeze()

            out = self.linear_out(feature)

            out_result.append(out)

        out_result = torch.cat(out_result, dim = 1)
                
        out_result = F.softmax(out_result, dim = 1)
        loss = criterion(out_result, labels)
        accu, acc_result = calc_accuracy(out_result, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(out_result, dim=1)[1].cpu().numpy(), "x": out_result, "accuracy_result": acc_result}


                

