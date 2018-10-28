import json
import torch
import numpy as np


class AYPredictionFormatter:
    labelToId = {}
    idToLabel = {}

    def __init__(self, config):
        label_list_file = config.get('data', 'label_list_file')
        fin = open(label_list_file, 'r')
        for line in fin.readlines():
            labelName = line.strip()
            self.labelToId[labelName] = len(self.labelToId)
            self.idToLabel[len(self.labelToId) - 1] = labelName

    def check(self, data, config):
        data = json.loads(data)
        if ',' in data['WS']['QTXX']['AY'] or data['WS']['QTXX']['AY'] == "":
            return None

        return data

    def pad(self, text, length, transformer):
        ans = []
        for v in text:
            if len(ans) >= length:
                break
            ans.append(transformer.load(v))
        while len(ans) < length:
            ans.append(transformer.load('BLANK'))
        return ans

    def format(self, data, config, transformer, mode):
        ss = []
        title = []
        pjjg = []
        label = []
        for line in data:
            sstmp = [v[0] for v in line['WS']['SS']['@value']]
            '''
            if len(sstmp) > config.getInt('train', 'ss_text_length'):
                for i in range(config.getInt('train', 'ss_text_lenth') - 50):
                    text.append(transformer.load(sstmp[i]))
                for i in range(50):
                    text.append(transformer.load(sstmp[i - 50]))

            else:
                for v in sstmp:
                    text.aapend(transformer.load(v))
                while len(text) < config.getInt('train', 'ss_text_lenght'):
                    text.append(transformer.load('BLANK'))
            '''
            ss.append(self.pad(sstmp, config.getint('train', 'ss_text_length'), transformer))
            '''
            for v in sstmp:
                if len(text) < config.getInt('train', 'ss_text_length'):
                    text.append(transformer.load(v))
                else:
                    break
            while len(text) < config.getInt('train', 'ss_text_length'):
                text.append(transformer.load('BLANK'))

            ss.append(text)
            '''

            titletmp = [v[0] for v in line['WS']['QTXX']['TITLE']['@value']]
            title.append(self.pad(titletmp, config.getint('train', 'title_length'), transformer))
            '''
            for v in titletmp:
                if len(titleText) < config.getInt('train', 'title_length'):
                    titleText.append(transformer.load('v'))
                else:
                    break
            while len(titleText) < config.getInt('train', 'title_length'):
                titleText.append(transformer.load('BLANK'))
            title.append(titleText)
            '''

            pjjgtmp = [v[0] for v in line['WS']['PJJG']['@value']]
            pjjg.append(self.pad(pjjgtmp, config.getint('train', 'pjjg_length'), transformer))

            tmp = np.zeros(len(self.labelToId))
            try:
                tmp[self.labelToId[line['WS']['QTXX']['AY']['@value']]] = 1
            except Exception as e:
                pass
            label.append(tmp)

        matrix = [ss[i] + title[i] + pjjg[i] for i in range(len(data))]
        matrix = torch.Tensor(matrix)
        label = torch.from_numpy(np.array(label, dtype=np.int))
        return {'input': matrix, 'label': label}
