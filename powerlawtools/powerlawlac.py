
class powerlawlac:
	def __init__(self,user_dict="/data/disk1/private/yx/THULAC/dict.txt", model_path="/home/yx/THULAC/models", T2S=False, seg_only=True, filt=False, max_length = 50000,deli='_',maxPerSent = 500):
		"""
		:param user_dict:/data/disk1/private/yx/THULAC/dict.txt
		:param model_path: 模型地址需要包含.so
		:param T2S:
		:param seg_only:
		:param filt:
		:param max_length:
		:param deli:
		"""
		import thulac
		self.lac = thulac.thulac(user_dict=user_dict,model_path=model_path,T2S = T2S,seg_only = seg_only,filt = filt,max_length = max_length,deli = deli)
		self.maxLOneS = maxPerSent
	def Lseg(self,sentence,endOFsentence = set(["。"])):
		ls = len(sentence)
		idx = 0
		while (idx<ls):
			ends = idx + self.maxLOneS
			if ends>=ls:
				ends = ls - 1
			while ends > idx + self.maxLOneS/2:
				# print(ends,ls)
				if sentence[ends] in endOFsentence:
					break
				ends -= 1
			ends += 1
			yield sentence[idx : ends]
			idx = ends

	def cut(self,sentence):
		res = []
		for seg in self.Lseg(sentence=sentence):
			res += self.lac.cut(seg)
		return res
	def fast_cut(self,sentence):
		res = []
		for seg in self.Lseg(sentence=sentence):
			res += self.lac.fast_cut(seg)
		return res
