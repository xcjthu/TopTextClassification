from dataloader import LineProcess,BatchProcess,FileReader,Folder2Data
lineprocess = LineProcess()
filereader = FileReader("/data/disk3/data/wenshu_data/json_data/zk/txt")
folder2data = Folder2Data(lineprocess=lineprocess,reader=filereader,batchsize=1,threadNum = 20 ,Qmax = 200)
res = 0
import numpy as np
for epoch in range(4):
	folder2data.start()
	data = folder2data.getData()
	while not data is None:
		res += np.sum(data["data"])
		data = folder2data.getData()
print(res)