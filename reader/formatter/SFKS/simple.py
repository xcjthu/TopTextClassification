import json
import torch


class SFKSSimpleAndEffectiveFormatter:
    def __init__(self, config):
        self.word2id = json.load(open(config.get("data", "word2id"), "r"))

        self.max_len = config.getint("data", "max_len")
        self.k = config.getint("data", "topk")

    def check(self, data, config):
        data = json.loads(data)
        if not ("answer" in data.keys()):
            return None
        if not (config.getboolean("data", "multi_choice")) and len(data["answer"]) != 1:
            return None
        return data

    def transform(self, word, transformer):
        if not (word in self.word2id.keys()):
            return self.word2id["UNK"]
        else:
            return self.word2id[word]

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

            result.append(self.transform(word, None))

        while len(result) < self.max_len:
            result.append(self.transform("PAD", None))

        return torch.LongTensor(result[0:self.max_len])

    def format(self, data, config, transformer, mode):
        question = []
        article = []
        label = []

        for temp_data in data:
            if config.getboolean("data", "multi_choice"):
                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x += 1
                if "B" in temp_data["answer"]:
                    label_x += 2
                if "C" in temp_data["answer"]:
                    label_x += 4
                if "D" in temp_data["answer"]:
                    label_x += 8
            else:
                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x = 0
                if "B" in temp_data["answer"]:
                    label_x = 1
                if "C" in temp_data["answer"]:
                    label_x = 2
                if "D" in temp_data["answer"]:
                    label_x = 3

            temp_question = []
            temp_article = []
            for option in ["A", "B", "C", "D"]:
                temp_question.append(self.parse(temp_data["statement"] + temp_data["option_list"][option]))

                temp = []
                for a in range(0, self.k):
                    temp.append(self.parse(temp_data["reference"][option][a]))

                temp_article.append(torch.stack(temp))

            question.append(torch.stack(temp_question))
            article.append(torch.stack(temp_article))
            label.append(label_x)

        label = torch.tensor(label, dtype=torch.long)
        # for x in question:
        #    print(x.size())
        question = torch.stack(question)
        article = torch.stack(article)

        return {
            "question": question,
            "article": article,
            "label": label
        }
