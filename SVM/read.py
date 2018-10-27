import json
import os
from sklearn.model_selection import train_test_split


data_dir = '/data/disk1/private/zhx/law/new_cutted_data/民事案件'

def read():
	filelist = os.listdir(data_dir)
	
	corpus = []
	labels = []
	
	count = 0
	for name in filelist:
		if name[-3:] != 'txt':
			continue
		count += 1
		if (count == 201):
			break
		print(name)
		fin = open(os.path.join(data_dir, name), 'r')
		datas = [json.loads(line) for line in fin.readlines()]
		for data in datas:
			ay = data['WS']['QTXX']['AY']['@value']
			if ',' in ay or ay == "":
				continue
			words = ' '.join([w[0] for w in data['WS']['PJJG']['@value']]) + ' '.join([w[0] for w in data['WS']['SS']['@value']])
			corpus.append(words)
			labels.append(ay)
		if len(corpus) > 300000:
			break
		fin.close()
		
	print(len(corpus))
	print(len(set(labels)))
	return corpus, labels


def getRunDataset():
	corpus, labels = read()
	return train_test_split(corpus, labels, test_size = 0.1, random_state = 0)

