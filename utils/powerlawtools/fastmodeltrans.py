import fasttext
import pickle
model = fasttext.load_model('/data/disk1/private/yx/model200v2_8.bin', encoding='utf-8')
(wordnum,vec_size) = (len(model.words),model.dim)
word2id = {}
vecList = []
for idx,word in enumerate(model.words):
    word2id[word] = idx
    vecList.append(model[word])
with open("/data/disk1/private/yx/word2id.pkl","wb") as f:
    pickle.dump((wordnum,vec_size),f)
    pickle.dump(word2id,f)
import numpy as np
vecnp = np.asarray(vecList)
print(vecnp.shape)
np.save("/data/disk1/private/yx/vec_nor.npy",vecnp)
