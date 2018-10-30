import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import calc_accuracy, print_info


class TextCNN(nn.Module):
	def __init__(self, config):
		super(TextCNN, self).__init__()
		self.data_size = config.getint("data", "vec_size")
		self.output_dim = config.getint("model", "output_dim")
		self.batch_size = config.getint('train', 'batch_size')
		
		self.min_gram = config.getint("model", "min_gram")
		self.max_gram = config.getint("model", "max_gram")
		self.convs = []
		for a in range(self.min_gram, self.max_gram + 1):
			self.convs.append(nn.Conv2d(1, config.getint('model', 'filters'), (a, self.data_size)))
		
		self.convs = nn.ModuleList(self.convs)
		self.feature_len = (self.max_gram - self.min_gram + 1) * config.getint('model', 'filters')
		self.fc = nn.Linear(self.feature_len, self.output_dim)
		self.relu = nn.ReLU()
	
	def init_hidden(self, config, usegpu):
		pass
	
	def init_multi_gpu(self, device):
		# self.convs = [nn.DataParallel(v) for v in self.convs]
		for conv in self.convs:
			conv = nn.DataParallel(conv)
		# self.convs = nn.ModuleList(self.convs)
		self.fc = nn.DataParallel(self.fc)
		self.relu = nn.DataParallel(self.relu)
		# pass

	def forward(self, data, criterion, config, usegpu):
		x = data['input']
		labels = data['label']

		x = x.view(self.batch_size, 1, -1, self.data_size)
		# print(x, labels)
		
		conv_out = []
		gram = self.min_gram
		# self.attention = []
		for conv in self.convs:
			y = self.relu(conv(x))#.view(config.getint("train", "batch_size"), config.getint("model", "filters"), -1)
			# self.attention.append(F.pad(y, (0, gram - 1)))
			# print(y.shape)
			y = torch.max(y, dim = 2)[0].view(self.batch_size, -1)
			# y = F.max_pool1d(y, kernel_size = x.shape[2]).view(self.batch_size, -1)

			conv_out.append(y)
			gram += 1

		conv_out = torch.cat(conv_out, dim = 1)
		# self.attention = torch.cat(self.attention, dim=1)

		y = self.fc(conv_out)
		
		loss = criterion(y, labels)
		accu = calc_accuracy(y, labels, config)
		return {"loss": loss, "accuracy": accu, "result": torch.max(y, dim=1)[1].cpu().numpy(), "x": y}
