import json
import torch
import numpy as np
import jieba
import random

from utils.util import check_multi


class ComatchingFormatter:
    def __init__(self, config):
        self.need = config.getboolean("data", "need_word2vec")
        if self.need:
            self.word_dim = config.getint("data", "vec_size")
        else:
            self.word2id = json.load(open(config.get("data", "word2id"), "r"))

        self.sent_max_len = config.getint("data", "sent_max_len")
        self.max_sent = config.getint("data", "max_sent")

        self.symbol = [",", ".", "?", "\"", "”", "。", "？", ""]
        self.last_symbol = [".", "?", "。", "？"]

    def check(self, data, config):
        data = json.loads(data)
        if not (config.getboolean("data", "multi_choice")) and len(data["answer"]) != 1:
            return None
        return data

    def transform(self, word, transformer):
        if not (word in self.word2id.keys()):
            if self.need:
                return transformer.load("UNK")
            else:
                return self.word2id["UNK"]
        else:
            if self.need:
                return transformer.load(word)
            else:
                return self.word2id[word]

    def seq2tensor(self, sents, max_len, transformer=None):
        sent_len_max = max([len(s) for s in sents])
        sent_len_max = min(sent_len_max, max_len)

        if self.need:
            sent_tensor = torch.FloatTensor(len(sents), sent_len_max, self.word_dim).zero_()
        else:
            sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()

        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max: break
                sent_tensor[s_id][w_id] = self.transform(word, transformer)
        return [sent_tensor, sent_len]

    def seq2Htensor(self, docs, max_sent, max_sent_len, transformer=None):
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, max_sent)
        sent_len_max = max([len(w) for s in docs for w in s])
        sent_len_max = min(sent_len_max, max_sent_len)

        if self.need:
            sent_tensor = torch.FloatTensor(len(docs), sent_num_max, sent_len_max, self.word_dim).zero_()
        else:
            sent_tensor = torch.LongTensor(len(docs), sent_num_max, sent_len_max).zero_()
        sent_len = torch.LongTensor(len(docs), sent_num_max).zero_()
        doc_len = torch.LongTensor(len(docs)).zero_()
        for d_id, doc in enumerate(docs):
            doc_len[d_id] = len(doc)
            for s_id, sent in enumerate(doc):
                if s_id >= sent_num_max: break
                sent_len[d_id][s_id] = len(sent)
                for w_id, word in enumerate(sent):
                    if w_id >= sent_len_max: break
                    sent_tensor[d_id][s_id][w_id] = self.transform(word, transformer)
        return [sent_tensor, doc_len, sent_len]

    def parse(self, sent):
        result = []
        for word in sent:
            if len(word) == 0:
                continue

            result.append(word)

        return result

    def parseH(self, sent):
        result = []
        temp = []
        for word in sent:
            temp.append(word)
            last = False
            for symbol in self.last_symbol:
                if word == symbol:
                    last = True
            if last:
                result.append(temp)
                temp = []

        if len(temp) != 0:
            result.append(temp)

        return result

    def format(self, data, config, transformer, mode):
        document = []
        option = []
        question = []
        label = []

        for temp_data in data:
            question.append(self.parse(temp_data["statement"]))

            if config.getboolean("data", "multi_choice"):
                option.append([self.parse(temp_data["option_list"]["A"]),
                               self.parse(temp_data["option_list"]["B"]),
                               self.parse(temp_data["option_list"]["C"]),
                               self.parse(temp_data["option_list"]["D"])])

                label_x = [0, 0, 0, 0]
                if "A" in temp_data["answer"]:
                    label_x[0] = 1
                if "B" in temp_data["answer"]:
                    label_x[1] = 1
                if "C" in temp_data["answer"]:
                    label_x[2] = 1
                if "D" in temp_data["answer"]:
                    label_x[3] = 1

                document.append(self.parseH(temp_data["analyse"]))
            else:
                option.append([self.parse(temp_data["option_list"]["A"]),
                               self.parse(temp_data["option_list"]["B"]),
                               self.parse(temp_data["option_list"]["C"]),
                               self.parse(temp_data["option_list"]["D"])])

                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x = 0
                if "B" in temp_data["answer"]:
                    label_x = 1
                if "C" in temp_data["answer"]:
                    label_x = 2
                if "D" in temp_data["answer"]:
                    label_x = 3

                document.append(self.parseH(temp_data["analyse"]))

            label.append(label_x)

        document = self.seq2Htensor(document, self.max_sent, self.sent_max_len, transformer)
        option = self.seq2Htensor(option, self.max_sent, self.sent_max_len, transformer)
        question = self.seq2tensor(question, self.sent_max_len, transformer)

        label = torch.tensor(label, dtype=torch.long)

        return {
            "question": question[0],
            "question_len": question[1],
            "option": option[0],
            "option_sent": option[1],
            "option_len": option[2],
            "document": document[0],
            "document_sent": document[1],
            "document_len": document[2],
            "label": label
        }


class ComatchingFormatter2:
    def __init__(self, config):
        self.need = config.getboolean("data", "need_word2vec")
        if self.need:
            self.word_dim = config.getint("data", "vec_size")
        else:
            self.word2id = json.load(open(config.get("data", "word2id"), "r"))

        self.sent_max_len = config.getint("data", "sent_max_len")
        self.max_sent = config.getint("data", "max_sent")

        self.symbol = [",", ".", "?", "\"", "”", "。", "？", ""]
        self.last_symbol = [".", "?", "。", "？"]

    def check(self, data, config):
        data = json.loads(data)
        if not (config.getboolean("data", "multi_choice")) and len(data["answer"]) != 1:
            return None
        return data

    def transform(self, word, transformer):
        if not (word in self.word2id.keys()):
            if self.need:
                return transformer.load("UNK")
            else:
                return self.word2id["UNK"]
        else:
            if self.need:
                return transformer.load(word)
            else:
                return self.word2id[word]

    def seq2tensor(self, sents, max_len, transformer=None):
        sent_len_max = max([len(s) for s in sents])
        sent_len_max = min(sent_len_max, max_len)

        if self.need:
            sent_tensor = torch.FloatTensor(len(sents), sent_len_max, self.word_dim).zero_()
        else:
            sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()

        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max: break
                sent_tensor[s_id][w_id] = self.transform(word, transformer)
        return [sent_tensor, sent_len]

    def seq2Htensor(self, docs, max_sent, max_sent_len, transformer=None, v1=0, v2=0):
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, max_sent)
        sent_len_max = max([len(w) for s in docs for w in s])
        sent_len_max = min(sent_len_max, max_sent_len)
        sent_num_max = max(sent_num_max, v1)
        sent_len_max = max(sent_len_max, v2)

        if self.need:
            sent_tensor = torch.FloatTensor(len(docs), sent_num_max, sent_len_max, self.word_dim).zero_()
        else:
            sent_tensor = torch.LongTensor(len(docs), sent_num_max, sent_len_max).zero_()
        sent_len = torch.LongTensor(len(docs), sent_num_max).zero_()
        doc_len = torch.LongTensor(len(docs)).zero_()
        for d_id, doc in enumerate(docs):
            doc_len[d_id] = len(doc)
            for s_id, sent in enumerate(doc):
                if s_id >= sent_num_max: break
                sent_len[d_id][s_id] = len(sent)
                for w_id, word in enumerate(sent):
                    if w_id >= sent_len_max: break
                    sent_tensor[d_id][s_id][w_id] = self.transform(word, transformer)
        return [sent_tensor, doc_len, sent_len]

    def gen_max(self, docs, max_sent, max_sent_len):
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, max_sent)
        sent_len_max = max([len(w) for s in docs for w in s])
        sent_len_max = min(sent_len_max, max_sent_len)

        return sent_num_max, sent_len_max

    def parse(self, sent):
        result = []
        for word in sent:
            if len(word) == 0:
                continue

            result.append(word)

        return result

    def parseH(self, sent):
        result = []
        temp = []
        for word in sent:
            temp.append(word)
            last = False
            for symbol in self.last_symbol:
                if word == symbol:
                    last = True
            if last:
                result.append(temp)
                temp = []

        if len(temp) != 0:
            result.append(temp)

        return result

    def format(self, data, config, transformer, mode):
        document = [[], [], [], []]
        option = []
        question = []
        label = []

        for temp_data in data:
            question.append(self.parse(temp_data["statement"]))

            if config.getboolean("data", "multi_choice"):
                option.append([self.parse(temp_data["option_list"]["A"]),
                               self.parse(temp_data["option_list"]["B"]),
                               self.parse(temp_data["option_list"]["C"]),
                               self.parse(temp_data["option_list"]["D"])])

                label_x = [0, 0, 0, 0]
                if "A" in temp_data["answer"]:
                    label_x[0] = 1
                if "B" in temp_data["answer"]:
                    label_x[1] = 1
                if "C" in temp_data["answer"]:
                    label_x[2] = 1
                if "D" in temp_data["answer"]:
                    label_x[3] = 1

                document.append(self.parseH(temp_data["analyse"]))
            else:
                option.append([self.parse(temp_data["option_list"]["A"]),
                               self.parse(temp_data["option_list"]["B"]),
                               self.parse(temp_data["option_list"]["C"]),
                               self.parse(temp_data["option_list"]["D"])])

                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x = 0
                if "B" in temp_data["answer"]:
                    label_x = 1
                if "C" in temp_data["answer"]:
                    label_x = 2
                if "D" in temp_data["answer"]:
                    label_x = 3

                temp = []
                for a in range(0, 4):
                    document[a].append(self.parseH(temp_data["analyse"]))

            label.append(label_x)

        v1 = 0
        v2 = 0
        for a in range(0, 4):
            v1t, v2t = self.gen_max(document[a], self.max_sent, self.sent_max_len)
            v1 = max(v1, v1t)
            v2 = max(v2, v2t)

        for a in range(0, 4):
            document[a] = self.seq2Htensor(document[a], self.max_sent, self.sent_max_len, transformer, v1, v2)
        option = self.seq2Htensor(option, self.max_sent, self.sent_max_len, transformer)
        question = self.seq2tensor(question, self.sent_max_len, transformer)

        document_sent = torch.stack([document[0][1], document[1][1], document[2][1], document[3][1]])
        document_len = torch.stack([document[0][2], document[1][2], document[2][2], document[3][2]])
        document = torch.stack([document[0][0], document[1][0], document[2][0], document[3][0]])
        document = torch.transpose(document, 0, 1)
        document_len = torch.transpose(document_len, 0, 1)
        document_sent = torch.transpose(document_sent, 0, 1)

        label = torch.tensor(label, dtype=torch.long)

        return {
            "question": question[0],
            "question_len": question[1],
            "option": option[0],
            "option_sent": option[1],
            "option_len": option[2],
            "document": document,
            "document_sent": document_sent,
            "document_len": document_len,
            "label": label
        }
