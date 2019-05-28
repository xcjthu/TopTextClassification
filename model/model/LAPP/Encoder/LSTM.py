import torch
import torch.nn as nn
import torch.functional as F


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        
        self.t = nn.Parameter(torch.Tensor(1, config.getint('model', 'hidden_size')))
        torch.nn.init.xavier_normal(self.t, gain=1)


    def forward(self, seq):
        out = seq.matmul(self.t.squeeze(0))
        out = torch.softmax(out, dim = 1).unsqueeze(1)
        return torch.bmm(out, seq).squeeze(1)


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.hidden_dim = config.getint("model", "hidden_size")

        self.lstm = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True,
                            num_layers=config.getint("model", "num_layers"))

        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.attention = Attention(config)

    def forward(self, data):
        x = data
        
        batch_size = x.shape[0]

        x = x.view(batch_size, -1, self.data_size)

        lstm_out, self.hidden = self.lstm(x)
        
        # lstm_out = torch.max(lstm_out, dim=1)[0]

        y = self.attention(lstm_out)

        return y
