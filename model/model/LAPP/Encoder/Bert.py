import torch
import torch.nn as nn
import torch.functional as F
from pytorch_pretrained_bert import BertModel


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("model", "hidden_size")

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.fc = nn.Linear(768, self.hidden_dim)


    def forward(self, data):
        x = data
        
        batch_size = x.shape[0]
        
        #print(x)
        #print(x.shape)
        _, y = self.bert(x, output_all_encoded_layers=False)

        y = y.view(y.size()[0], -1)
        
        y = self.fc(y)

        return y
