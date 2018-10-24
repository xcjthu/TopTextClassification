from fastvec import word2vec

from powerlawlac import powerlawlac

from time import sleep
import multiprocessing as mp
import numpy as np
import json
class FileReader:
	def __init__(self,folder):
		self.rootdir = folder
		self.filesList = None
		self.fileq = mp.Queue()
		self.fileHandle = None
		self.lock = mp.Lock()
	def getAllFiles(self,rootdir):
		import os
		filesList = []
		entityList = os.listdir(rootdir) #列出文件夹下所有的目录与文件
		for i in range(0,len(entityList)):
			path = os.path.join(rootdir,entityList[i])
			if os.path.isfile(path):
				filesList.append(path)
			if os.path.isdir(path):
				filesList += self.getAllFiles(path)
		return filesList
	def getFileHandle(self):
		self.lock.acquire()
		self.fileHandle = None
		if not self.fileq.empty():
			fileName = self.fileq.get()
			self.fileHandle = open(fileName)
		self.lock.release()
	def readOneline(self,):
		if (self.fileHandle is None):
			self.getFileHandle()
		data = self.fileHandle.readline()
		while data == '':
			self.getFileHandle()
			if (self.fileHandle is None):
				return None
			data = self.fileHandle.readline()
		return data
	def start(self):
		self.filesList = self.getAllFiles(self.rootdir)
		for fileName in self.filesList:
			self.fileq.put(fileName)


class LineProcess:
	def __init__(self,):
		self.wv = word2vec()
		self.segment = powerlawlac()
		# pass
	def process(self,line):
		data = json.loads(line)
		segs = [x[0] for x in self.segment.fast_cut(data["WS"]["QW"]["@value"])]
		res = np.zeros(200)
		for word in segs:
			res += self.wv.load(word)
		res /= len(segs)
		return {"data":res,"label":1}


class BatchProcess:
	def __init__(self,batchSize = 1,lineprocess = None,reader = None):
		self.batchsize = batchSize
		self.lineprocess = lineprocess
		self.reader = reader
	def nextBatch(self):
		res ={}
		for x in range(self.batchsize):
			data = self.reader.readOneline()
			# print(x,self.batchsize,data)
			if (data is None):
				return None
			data = self.lineprocess.process(data)
			for k in data:
				if k in res:
					res[k].append(data[k])
				else:
					res[k] = [ data[k] ]
		for k in res:
			res[k] = np.asarray(res[k])
		return res


class Folder2Data:
	def __init__(self,lineprocess,reader, batchsize = 2, threadNum = 1, Qmax = 2):
		self.dataq = mp.Queue()
		self.threadNum = threadNum
		self.fileHandle = None
		self.Qmax = Qmax
		self.lineProcess = lineprocess
		self.reader = reader
		# self.processer = processer
		self.none_cnt = 0
		self.batchsize = batchsize
		self.readProcess = []
		self.pool = mp.Pool(processes=self.threadNum)

	def readtoQueue(self,threadidx):

		self.reader.getFileHandle()
		if self.reader.fileHandle is None:
			return None
		batchprocess = BatchProcess(batchSize = self.batchsize, lineprocess = self.lineProcess, reader = self.reader)
		while (True):
			if self.dataq.qsize()<self.Qmax:
				data = batchprocess.nextBatch()
				self.dataq.put(data)
				if (data is None):
					break
			else:
				sleep(0.001)

	def start(self):
		self.none_cnt = 0
		self.reader.start()
		# self.readtoQueue(0)

		for idx in range(0, self.threadNum):
			process = mp.Process(target=self.readtoQueue,
											  args=(idx,))
			self.readProcess.append(process)
			self.readProcess[-1].start()

	def getData(self):
		while True:
			data = self.dataq.get()
			if data is None:
				self.none_cnt += 1
				if self.none_cnt == self.threadNum:
					self.none_cnt = 0
					break
			else:
				break

		return data