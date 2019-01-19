import json
import torch
import numpy as np
import jieba
import os
from pytorch_pretrained_bert import BertTokenizer


class SFKSBertSubjectFormatter:
    def __init__(self, config):
        self.map_list = doc_type_map_dic = {
            "国际法": 0,
            "刑法": 1,
            "刑事诉讼法【最新更新】": 2,
            "司法制度和法律职业道德": 3,
            "法制史": 4,
            "民法": 5,
            "民诉与仲裁【最新更新】": 6,
            "国际经济法": 7,
            "法理学": 8,
            "国际私法": 9,
            "社会主义法治理念": 10,
            "商法": 11,
            "民诉与仲裁【更新中】": 6,
            "行政法与行政诉讼法": 12,
            "宪法": 13,
            "经济法": 14
        }
        self.max_len = config.getint("data", "max_len")

        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(config.get("model", "bert_path"), "vocab.txt"))

    def check(self, data, config):
        data = json.loads(data)
        if len(data["statement"]) == 0:
            return None
        if not ("subject" in data.keys()):
            return None
        return data

    def convert(self, tokens):
        ids = []
        for token in tokens:
            if '\u4e00' <= token <= '\u9fff':
                if token in self.tokenizer.vocab.keys():
                    ids.append(self.tokenizer.vocab[token])
                else:
                    # print("<<<<<<<<<<<<< %s >>>>>>>>>>>>> " % token)
                    ids.append(self.tokenizer.vocab["[UNK]"])

        while len(ids) < self.max_len:
            ids.append(self.tokenizer.vocab["[PAD]"])
        return ids

    def format(self, data, config, transformer, mode):
        input = []
        label = []
        for temp_data in data:
            ss = []
            res = temp_data["statement"] + temp_data["option_list"]["A"] + temp_data["option_list"]["B"] + \
                  temp_data["option_list"]["C"] + temp_data["option_list"]["D"]
            for a in range(0, len(res)):
                ss = ss + [res[a]]
            ss = ss[0:self.max_len]

            indexed_tokens = self.convert(ss)

            tokens_tensor = torch.tensor([indexed_tokens])

            input.append(tokens_tensor)
            label.append(self.map_list[temp_data["subject"]])

        input = torch.cat(input, dim=0)
        label = torch.LongTensor(np.array(label, dtype=np.int32))
        return {'input': input, 'label': label}
