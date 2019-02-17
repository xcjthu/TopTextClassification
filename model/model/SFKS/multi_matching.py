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

        self.vecsize = config.getint('data', 'vec_size')
        #self.hidden_size = config.getint('model', 'hidden_size')
        self.hidden_size = self.vecsize // 2

        bidirectional = True
        self.gru = nn.GRU(self.vecsize, self.hidden_size, batch_first = True, bidirectional = bidirectional)
        
        self.wh = nn.Linear(self.vecsize, self.vecsize, bias = False)
        self.we = nn.Linear(self.vecsize, self.vecsize)

        self.init_para()


    def init_para(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


    def forward(self, in_seq):
        gru_out, gru_hidden = self.gru(in_seq)

        out = self.wh(gru_out) + self.we(in_seq)

        reset_gate = F.sigmoid(out)

        # reset_gate = reset_gate.expand(in_seq.shape[0], in_seq.shape[1], in_seq.shape[2])
        return in_seq.mul(reset_gate) + gru_out.mul(1 - reset_gate)


class SoftSel(nn.Module):
    def __init__(self, config):
        super(SoftSel, self).__init__()

        self.hidden_size = config.getint('data', 'vec_size')

        self.wg = nn.Parameter(Var(torch.Tensor(self.hidden_size, self.hidden_size)))
        torch.nn.init.xavier_uniform_(self.wg, gain=1)
        
    def forward(self, hi1, hi2):
        G = hi1.matmul(self.wg)
        G = torch.bmm(G, torch.transpose(hi2, 1, 2))
        #G = torch.bmm(hi1, self.wg.unsqueeze(0).expand(hi1.shape[0], self.hidden_size, self.hidden_size))
        #G = torch.bmm(G, torch.transpose(hi2, 1, 2))
        
        G = F.softmax(G, dim = 2)
        return torch.bmm(G, hi2)


class Match(nn.Module):
    def __init__(self, config):
        super(Match, self).__init__()
        
        self.vecsize = config.getint('data', 'vec_size')
        self.hidden_size = config.getint('data', 'vec_size')

        self.match = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.linear_hm = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.match_gru = nn.GRU(self.hidden_size, self.hidden_size // 2, batch_first = True, bidirectional = True)
        
        self.linear_att = nn.Linear(self.hidden_size, 1)


    def forward(self, hi1, hi2, hi3, hi4):
        m1 = torch.cat([hi1 + hi2, hi1 - hi2], dim = 2)
        m2 = torch.cat([hi3 + hi4, hi3 - hi4], dim = 2)

        m1 = F.relu(self.match(m1))
        m2 = F.relu(self.match(m2))

        hm = self.linear_hm(torch.cat([m1, m2], dim = 2))

        hm, match_hidden = self.match_gru(hm)
        
        hmax, _ = torch.max(hm, dim = 1)
        
        hatt_softmax = F.softmax(F.relu(self.linear_att(hm)), dim = 1).squeeze().unsqueeze(1)
        hatt = torch.bmm(hatt_softmax, hm).squeeze()

        hf = torch.cat([hmax, hatt], dim = 1)
        return hf


class EAMatch(nn.Module):
    def __init__(self, config):
        super(EAMatch, self).__init__()


        self.match = Match(config)

        '''
        self.softsel_QP = SoftSel(config)
        self.softsel_AQ = SoftSel(config)
        self.softsel_QAA = SoftSel(config)
        self.softsel_AQA = SoftSel(config)
        '''
        self.softsel = SoftSel(config)
        

    def forward(self, hq, hp, ha):
        '''
        hq_ = self.softsel_QP(hq, hp)
        hqa = self.softsel_AQ(ha, hq_)
        hqa_ = self.softsel_QAA(hqa, ha)
        ha_ = self.softsel_AQA(ha, hqa)
        '''
        hq_ = self.softsel(hq, hp)
        hqa = self.softsel(ha, hq_)
        hqa_ = self.softsel(hqa, ha)
        ha_ = self.softsel(ha, hqa)
        

        hf = self.match(hqa, hqa_, ha, ha_)

        return hf


class QPAMatch(nn.Module):
    def __init__(self, config):
        super(QPAMatch, self).__init__()
        
        '''
        self.softsel_PQ = SoftSel(config)
        self.softsel_PA = SoftSel(config)
        '''
        self.softsel = SoftSel(config)
        

        self.match = Match(config)

    def forward(self, hq, hp, ha):
        '''
        hpq = self.softsel_PQ(hp, hq)
        hpa = self.softsel_PA(hp, ha)
        '''
        hpq = self.softsel(hp, hq)
        hpa = self.softsel(hp, ha)
        
        #print('\n\n\n', self.softsel_PQ.wg)

        hf3 = self.match(hpq, hp, hpa, hp)
        return hf3


class GateLayer(nn.Module):
    def __init__(self, config, feature_size):
        super(GateLayer, self).__init__()

        self.topN = config.getint('data', 'topN')
        self.linear = nn.Linear(feature_size, 1)
    
    def forward(self, all_doc_out):
        # batchsize * doc_num * feature_size
        out = self.linear(all_doc_out).squeeze(2)
        out = F.relu(out) + 0.0001
        out = out.mul(out)
        sumof = torch.sum(out, dim = 1)
        out = (out / sumof.unsqueeze(1))

        #print(out.shape)
        #print(all_doc_out.shape)

        feature = torch.bmm(out.unsqueeze(1), all_doc_out)
        
        return feature, all_doc_out



class OutputLayer(nn.Module):
    def __init__(self, config, input_dim):
        super(OutputLayer, self).__init__()
        
        self.output_strategy = config.get('model', 'output_strategy')
        
        self.linear = nn.Linear(input_dim, 1)
        if self.output_strategy == "fc_fc":
            self.linear2 = nn.Linear(config.getint('data', 'topN'), 1)

        self.multi = config.getboolean('data', 'multi_choice')
        if self.multi:
            self.mutli_module = nn.Linear(4, 16)


    def forward(self, feature):
        # feature size: (batch, 4, doc_num, input_dim)
        #print('feature size:', feature.shape)
        scores = []
        #for i in range(4):
        #feat = feature[:,i].contiguous()
        feat = feature.view(feature.shape[0] * feature.shape[1], feature.shape[2], feature.shape[3])
        if self.output_strategy == "max_pooling_fc":
            feat, _ = torch.max(feat, dim = 1)
            feat = self.linear(feat)
            
        elif self.output_strategy == "fc_fc":
            feat = self.linear(feat).squeeze(2)
            feat = self.linear2(feat)
            scores.append(feat)
            
        elif self.output_strategy == "fc_max_pooling":
            feat = self.linear(feat)
            feat, _ = torch.max(feat, dim = 1)
            
        #scores.append(feat)
        
        #scores = torch.cat(scores, dim = 1)
        scores = feat.squeeze().view(-1, 4)
        if self.multi:
            scores = self.multi_module(scores)
        
        return scores
            


class MultiMatchNet(nn.Module):
    def __init__(self, config):
        super(MultiMatchNet, self).__init__()

        self.hidden_size = config.getint('data', 'vec_size')
        
        self.emb_dim = config.getint('data', 'vec_size')
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))

        self.embs = nn.Embedding(self.word_num, self.emb_dim)
        #if config.getboolean("data", "need_word2vec"):
        #    self.embs = generate_embedding(self.embs, config)


        self.topN = config.getint('data', 'topN')
        self.batch_size = config.getint('train', 'batch_size')


        self.input = InputLayer(config)
        self.EAM1 = EAMatch(config)
        #self.EAM2 = EAMatch(config)

        self.QPAM = QPAMatch(config)
        
        #self.output_linear0 = nn.Linear(6 * self.hidden_size, 2 * self.hidden_size)
        #self.output_linear1 = nn.Linear(2 * self.hidden_size, 1)
        
        #if self.need_gate:
        #    self.out_gate = GateLayer(config, 6 * self.hidden_size)

        self.output = OutputLayer(config, self.hidden_size * 6)


    def init_multi_gpu(self, device):
        self.EAM1 = nn.DataParallel(self.EAM1)
        self.QPAM = nn.DataParallel(self.QPAM)
        #self.output_linear0 = nn.DataParallel(self.output_linear0)
        #self.output_linear1 = nn.DataParallel(self.output_linear1)
        
        #if self.need_gate:
        #    self.out_gate = nn.DataParallel(self.out_gate)
        
        self.input = nn.DataParallel(self.input)
        self.embs = nn.DataParallel(self.embs)

        self.output = nn.DataParallel(self.output)


    def forward(self, data, criterion, config, usegpu, acc_result = None):
        question = data['statement']
        option_list = data['answer']
        documents = data['reference']
        labels = data['label']

        '''
        print('input')
        print('question.shape', question.shape)
        print('option_list.shape', option_list.shape)
        print('document.shape', documents.shape)
        print('labels', labels.shape)
        '''


        if not config.getboolean("data", "need_word2vec"):
            question = self.embs(question)
        
        hq = self.input(question)
        hq = hq.unsqueeze(1).repeat(1, 4, 1, 1)
        hq = hq.view(self.batch_size * 4, hq.shape[2], hq.shape[3])
        hq = hq.unsqueeze(1).repeat(1, self.topN, 1, 1)
        hq = hq.view(self.batch_size * 4 * self.topN, hq.shape[2], hq.shape[3])




        out = []
        feature = []

        #for option_index in range(4):
        #    option = option_list[:,option_index].contiguous()
        #    docs = documents[:,option_index].contiguous()
        option = option_list.view(self.batch_size * 4, option_list.shape[2], option_list.shape[3])
        docs = documents.view(self.batch_size * 4, documents.shape[2], documents.shape[3], documents.shape[4])
            
        if not config.getboolean('data', 'need_word2vec'):
            option = self.embs(option)

        ha = self.input(option)

        ha = ha.unsqueeze(1).repeat(1, self.topN, 1, 1)
        ha = ha.view(self.batch_size * 4 * self.topN, ha.shape[2], ha.shape[3])
            
        hfs = []
        '''
            for doc_index in range(docs.shape[1]):
                doc = docs[:,doc_index].contiguous()

                #取第一个文章
                #doc = docs[:,0].contiguous()

                if not config.getboolean('data', 'need_word2vec'):
                    doc = self.embs(doc)

                hp = self.input(doc)
                # hq = self.input(question)
                # ha = self.input(option)

                hf1 = self.EAM1(hq, hp, ha)
                hf2 = self.EAM1(hq, ha, hp)
                hf3 = self.QPAM(hq, hp, ha)

                hf = torch.cat([hf1, hf2, hf3], dim = 1)
                
                # hf: 6 * hidden_size
                hfs.append(hf.unsqueeze(1))

            hfs = torch.cat(hfs, dim = 1) # size: (batch, doc_num, 6 * hidden_size)
        '''
            
        doc = docs.view(self.batch_size * self.topN * 4, docs.shape[2], docs.shape[3])
            
        if not config.getboolean('data', 'need_word2vec'):
            doc = self.embs(doc)

        hp = self.input(doc)
        hf1 = self.EAM1(hq, hp, ha)
        hf2 = self.EAM1(hq, ha, hp)
        hf3 = self.QPAM(hq, hp, ha)

        hfs = torch.cat([hf1, hf2, hf3], dim = 1)
        hfs = hfs.view(self.batch_size * 4, self.topN, 6 * self.hidden_size)
        
        feature = hfs.view(self.batch_size, 4, self.topN, 6 * self.hidden_size)

        #feature.append(hfs.unsqueeze(1))

        #feature = torch.cat(feature, dim = 1)
        out_result = self.output(feature)

        # out_result = torch.cat(out, dim = 1)
        
        loss = criterion(out_result, labels)
        accu, acc_result = calc_accuracy(out_result, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(out_result, dim=1)[1].cpu().numpy(), "x": out_result, "accuracy_result": acc_result}



