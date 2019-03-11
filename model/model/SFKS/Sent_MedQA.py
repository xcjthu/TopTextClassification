import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as Var
import numpy as np
import json

from utils.util import calc_accuracy, gen_result


class MultiScaleContext(nn.Module):
    def __init__(self, config):
        super(MultiScaleContext, self).__init__()

    def forward(self, ):
        pass


class ContextAttention(nn.Module):
    def __init__(self, config):
        super(ContextAttention, self).__init__()
        
        self.hidden_size = config.getint('model', 'hidden_size')

        self.ws1 = nn.Parameter(self.hidden_size * 2, self.hidden_size) #nn.Linear(self.hidden_size * 2, self.hidden_size, bias = False)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first = True, bidirectional = True)
        self.ws2 = nn.Linear(self.hidden_size * 2, self.hidden_size)


    def forward(self, in_seq):
        out = in_seq.matmul(self.ws1)
        out = torch.tanh(out)
        out, hidden = self.lstm(out)
        out = out.matmul(self.ws2)
        out = torch.bmm(torch.transpose(out, 1, 2), in_seq)
        return out

class SementicMatchScore(nn.Module):
    def __init__(self, config):
        super(SementicMatchScore, self).__init__()

    def forward(self, )
