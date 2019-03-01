import torch
import torch.nn as nn
import json

from utils.util import calc_accuracy, print_info

torch.manual_seed(1)


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x


class FFZJDPCNN(nn.Module):
    """
    DPCNN model, 3
    1. region embedding: using TetxCNN to generte
    2. two 3 conv(padding) block
    3. maxpool->3 conv->3 conv with resnet block(padding) feature map: len/2
    """

    # max_features, opt.EMBEDDING_DIM, opt.SENT_LEN, embedding_matrix):
    def __init__(self, config):
        super(FFZJDPCNN, self).__init__()
        self.model_name = "DPCNN"
        self.emb_dim = config.getint("data", "vec_size")  # 300
        self.mem_dim = config.getint("model", "hidden_size")  # 150
        self.output_dim = config.getint("model", "output_dim")
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))

        self.embedding = nn.Embedding(self.word_num, self.emb_dim)

        # region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(self.emb_dim, self.mem_dim,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.mem_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=self.mem_dim),
            nn.ReLU(),
            nn.Conv1d(self.mem_dim, self.mem_dim,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.mem_dim),
            nn.ReLU(),
            nn.Conv1d(self.mem_dim, self.mem_dim,
                      kernel_size=3, padding=1),
        )

        self.num_seq = config.getint("data", "max_len")
        resnet_block_list = []
        while (self.num_seq > 2):
            resnet_block_list.append(ResnetBlock(self.mem_dim))
            self.num_seq = self.num_seq // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.fc = nn.Sequential(
            nn.Linear(self.mem_dim * self.num_seq, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.output_dim, self.output_dim)
        )

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        x = data['input']
        labels = data['label']

        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.region_embedding(x)
        x = self.conv_block(x)
        x = self.resnet_layer(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(config.getint("train","batch_size"), -1)
        out = self.fc(x)
        y = out

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
