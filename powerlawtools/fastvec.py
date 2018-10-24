import numpy as np
import pickle
import sharedmem
class word2vec:
	word_num = 0
	vec_len = 0
	word2id = None
	vec = None
	def __init__(self, word_dic="/data/disk1/private/yx/word2id_16.pkl",vec_path="/data/disk1/private/yx/vec_nor_16.npy"):
		"""
			:param word_dic:/data/disk1/private/yx/word2id.pkl，/data/disk1/private/zhx/law/word2vec/word2id.pkl
			:param vec_path: /data/disk1/private/yx/vec_nor.npy，/data/disk1/private/zhx/law/word2vec/vec_nor.npy
		"""
		print("begin to load word embedding")
		f = open(word_dic, "rb")
		(self.word_num, self.vec_len) = pickle.load(f)
		self.word2id = pickle.load(f)
		f.close()
		# vec = np.load(vec_path)
		self.vec = np.load(vec_path)#sharedmem.empty(vec.shape,dtype = 'f8')
		#self.vec[:,:] = vec[:,:]
		# del vec
		print("load word embedding succeed")

	def load(self, word):
		try:
			return self.vec[self.word2id[word]].astype(dtype=np.float32)
		except:
			return np.zeros_like(self.vec[0]).astype(dtype=np.float32)
