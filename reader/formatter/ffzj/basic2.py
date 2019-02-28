import json
import torch
import numpy as np
import jieba
import random
import os
from pytorch_pretrained_bert import BertTokenizer


class FFZJBasicFormatter2:
    def __init__(self, config):
        self.map_list = {}
        l = [
            "法院依职权作出非法证据调查",
            "认定-刑讯逼供（暴力）",
            "认定-其他不排除非法手段获取的有罪供述",
            "法院认为收集证人证言程序不合法",
            "一审法院对非法证据排除申请没有审查并且作为定案根据",
            "讯问笔录没有经被告人核对确认并签名（盖章）、捺指印",
            "讯问笔录填写的讯问时间、讯问人、记录人、法定代理人等有误或者存在矛盾，不能补正或者作出合理解释",
            "以暴力、威胁、非法限制人身自由等手段获取证言",
            "引诱、欺骗等非法方法收集证人证言",
            "暴力、威胁、非法限制人身自由等手段获得被害人陈述",
            "其他不符合法定程序收集物证",
            "其他不符合法定程序收集书证",
            "不符合法定程序收集视听资料、电子证据",
            "庭审前启动非法证据排除调查程序",
            "非法拘禁等非法限制人身自由的方法",
            "认定-未依法对讯问进行全程录音录像或者讯问笔录与录音录像不符",
            "认定-取得被告人供述的时间、地点、侦查主体等不符合法律规定",
            "取得证人证言的时间、地点、询问主体等不符合法律规定",
            "依法应当出庭作证的证人没有正当理由拒绝出庭或者出庭后拒绝作证",
            "讯问笔录的内容出现错误、遗漏、被替换等情况",
            "证人与案件有利害关系",
            "庭审前决定排除非法证据",
            "庭审中提出排除申请",
            "瑕疵不排除",
            "庭审中启动证据排除审查程序",
            "庭审中证据排除申请不予受理",
            "视为对证据排除申请不受理（法院未回应）",
            "证据类型不确定",
            "被告人申请非法证据排除",
            "申请时提供有关线索或材料",
        ]
        for a in range(0, len(l)):
            x = l[a]
            self.map_list[x] = a

        self.xd = {}
        xx = 30
        for a in range(0, 30):
            for b in range(a + 1, 30):
                xx += 1
                self.xd[(a, b)] = xx
        self.max_len = config.getint("data", "max_len")
        self.word2id = json.load(open(config.get("data", "word2id"), "r"))

    def check(self, data, config, mode):
        data = json.loads(data)
        if len(data["text"]) == 0:
            return None
        if len(data["label"]) == 0 and len(data["text"]) < 15:
            return None
        if len(data["label"]) == 0 and mode == "train":
            if random.randint(1, 100) != 1:
                return None
        return data

    def transform(self, word):
        if not (word in self.word2id.keys()):
            return self.word2id["UNK"]
        else:
            return self.word2id[word]

    def convert(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.transform(token))

        while len(ids) < self.max_len:
            ids.append(self.transform("PAD"))
        return ids

    def format(self, data, config, transformer, mode):
        input = []
        label = []
        for temp_data in data:
            res = temp_data["text"]
            res = res[0:self.max_len]

            indexed_tokens = self.convert(res)

            tokens_tensor = torch.tensor([indexed_tokens])

            input.append(tokens_tensor)

            labels = 0
            if len(temp_data["label"]) == 0:
                labels = 0
            elif len(temp_data["label"]) == 1:
                labels = self.map_list[temp_data["label"][0]] + 1
            else:
                a = self.map_list[temp_data["label"][0]]
                b = self.map_list[temp_data["label"][1]]
                if a > b:
                    a, b = b, a
                labels = self.xd[(a, b)]
            label.append(labels)

        input = torch.cat(input, dim=0)
        label = torch.LongTensor(np.array(label, dtype=np.int32))
        return {'input': input, 'label': label}
