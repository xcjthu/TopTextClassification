import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from utils.util import calc_accuracy, gen_result


class CNNEnocoder(nn.Module):
    def __init__(self, config, output_len, need_fc):
        super(CNNEnocoder, self).__init__()
        self.convs = []
        for a in range(config.getint("model", "min_gram"), config.getint("model", "max_gram") + 1):
            self.convs.append(
                nn.Conv2d(1, config.getint("model", "filters"), (a, config.getint("data", "vec_size"))))

        self.convs = nn.ModuleList(self.convs)

        self.feature_len = (-config.getint("model", "min_gram") + config.getint("model",
                                                                                "max_gram") + 1) * config.getint(
            "model", "filters")

        self.need_fc = need_fc
        self.fc = nn.Linear(self.feature_len, output_len)

    def forward(self, x, config):
        l = len(x[0])
        x = x.view(config.getint("train", "batch_size"), 1, -1, config.getint("data", "vec_size"))
        conv_out = []
        gram = config.getint("model", "min_gram")
        for conv in self.convs:
            y = F.relu(conv(x)).view(config.getint("train", "batch_size"), config.getint("model", "filters"), -1)
            y = F.max_pool1d(y, kernel_size=l - gram + 1).view(
                config.getint("train", "batch_size"), -1)
            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)
        feature = conv_out.view(-1, self.feature_len)

        if self.need_fc:
            feature = self.fc(feature)

        return feature


class BasicCNN(nn.Module):
    def __init__(self, config):
        super(BasicCNN, self).__init__()

        self.word_size = config.getint("data", "vec_size")
        self.word_num = len(json.load(open(config.get("data", "word2id"), "r")))
        self.output_dim = config.getint("model", "output_dim")

        self.statement_encoder = CNNEnocoder(config, self.output_dim, True)
        self.answer_encoder = CNNEnocoder(config, self.output_dim, True)

        self.embedding = nn.Embedding(self.word_num, self.word_size)
        self.bilinear = nn.Bilinear(self.output_dim, self.output_dim, 1)

        self.sigmoid = nn.Sigmoid()

        if config.get('train', 'type_of_loss') == 'multi_label_cross_entropy_loss':
            self.multi = True
        else:
            self.multi = False

    def init_multi_gpu(self, device):
        pass

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        statement = data['statement']
        answer = data["answer"]
        labels = data["label"]

        statement = self.embedding(statement)
        answer = self.embedding(answer)
        print(statement)

        statement = self.statement_encoder(statement)
        ans_list = []
        for a in range(0, 4):
            temp = answer[:, a]
            ans_list.append(self.bilinear(statement, temp))

        y = torch.Tensor(ans_list)
        y = self.sigmoid(y)

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)
        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
