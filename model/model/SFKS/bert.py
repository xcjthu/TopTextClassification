import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel

from utils.util import calc_accuracy, print_info


class SFKSBert(nn.Module):
    def __init__(self, config):
        super(SFKSBert, self).__init__()
        self.data_size = config.getint("data", "vec_size")
        self.output_dim = config.getint("model", "output_dim")
        self.batch_size = config.getint('train', 'batch_size')

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        if config.get("model", "rank_method") == "all":
            self.rank_module = nn.Linear(
                768 * (config.getint("data", "max_len1") + config.getint("data", "max_len2")) * config.getint("data",
                                                                                                              "topk"),
                1)
        else:
            self.rank_module = nn.Linear(768 * (config.getint("data", "max_len1") + config.getint("data", "max_len2")),
                                         1)

        self.sigmoid = nn.Sigmoid()

        if config.get('train', 'type_of_loss') == 'multi_label_cross_entropy_loss':
            self.multi = True
        else:
            self.multi = False

    def init_multi_gpu(self, device):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, criterion, config, usegpu, acc_result=None):
        text = data["text"]
        token = data["token"]
        mask = data["mask"]

        batch = text.size()[0]
        option = text.size()[1]
        k = config.getint("data", "topk")
        option = option // k

        # print(text.size())
        # print(token.size())
        # print(mask.size())

        text = text.view(text.size()[0] * text.size()[1], text.size()[2])
        token = token.view(token.size()[0] * token.size()[1], token.size()[2])
        mask = mask.view(mask.size()[0] * mask.size()[1], mask.size()[2])
        # print(text)
        # print(token)
        # print(mask)

        labels = data['label']

        # encode, y = self.bert.forward(text)
        y, encode = self.bert.forward(text, token, mask, output_all_encoded_layers=False)
        # print(encode.size())
        # print(y.size())

        if config.get("model", "rank_method") == "all":
            y = y.view(batch * option, -1)
            y = self.rank_module(y)

            y = y.view(batch, option)
        elif config.get("model", "rank_method") == "max":
            y = y.view(batch * option, k, -1)
            y = torch.max(y, dim=1)[0]
            y = y.view(batch * option, -1)
            y = self.rank_module(y)
            y = y.view(batch, option)
            output = y
        else:
            y = y.view(batch * option * k, -1)
            y = self.rank_module(y)
            y = y.view(batch, option, k)
            y = torch.max(y, dim=2)[0]

            y = y.view(batch, option)

        # print(y.size())

        # gg

        loss = criterion(y, labels)
        accu, acc_result = calc_accuracy(y, labels, config, acc_result)

        return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y,
                "accuracy_result": acc_result}
