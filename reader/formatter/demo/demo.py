import json
import torch



class DemoFormatter:
    def __init__(self, config):
        f = open(config.get('data', 'law_label'), 'r')
        self.law_label = [line.strip() for line in f.readlines()]
        f = open(config.get('data', 'charge_label'), 'r')
        self.charge_label = [line.strip() for line in f.readlines()]

        f = open(config.get('data', 'attribute_path'), 'r')
        self.attr = json.loads(f.read())
        self.attr_vec = [[1, 0], [0, 1], [0, 0]]
        
        self.max_len = config.getint('data', 'max_len')
        self.word2id = json.load(open(config.get("data", "word2id"), "r"))
    

    def lookup(self, passage, max_len):
        lookup_id = []
        for word in data:
            try:
                lookup_id.append(self.word2id[word])
            except:
                lookup_id.append(self.word2id["UNK"])

        while len(lookup_id) < max_len:
            lookup_id.append(self.word2id["PAD"])
        lookup_id = lookup_id[:max_len]

        return lookup_id



    def gettime(self, time):
        #将刑期用分类模型来做
        v = int(time['imprisonment'])

        if time['death_penalty']:
            return 0
        if time['life_imprisonment']:
            return 1
        elif v > 10 * 12:
            return 2
        elif v > 7 * 12:
            return 3
        elif v > 5 * 12:
            return 4
        elif v > 3 * 12:
            return 5
        elif v > 2 * 12:
            return 6
        elif v > 1 * 12:
            return 7
        else:
            return 8

    def getAttribute(self, charge):
        try:
            attr = self.attr[charge]
        except:
            attr = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        return [self.attr_vec[v] for v in attr]


    def check(self, data, config):
        pass

    def format(self, data, config, transformer, mode):
        label = {'law': [], 'charge': [], 'time': [], 'attribute': []}
        
        passage = []
        for line in data:
            law = self.law_label[line['meta']['relevant_articles'][0]]
            charge = self.charge_label[line['meta']['accusation'][0]]
            time = self.gettime(line['meta']['term_of_imprisonment'])
            attribute = self.getAttribute(line['meta']['accusation'][0])
        

            label['law'].append(law)
            label['charge'].append(charge)
            label['time'].append(time)
            label['attribute'].append(attribute)

            passage.append(self.lookup(line['fact'], self.max_len))
            

        
        label['law'] = torch.LongTensor(label['law'])
        label['charge'] = torch.LongTensor(label['charge'])
        label['time'] = torch.LongTensor(label['time'])
        label['attribute'] = torch.FloatTensor(label['attribute'])
        
        passage = torch.LongTensor(passage)

        return {'docs': passage, 'label': label}








