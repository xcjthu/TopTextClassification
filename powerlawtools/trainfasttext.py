import fasttext
import pickle
base = 1
destDir ="/data/disk1/private/yx/"
while (1):
    desFile = destDir+'model'+str(base)
    print(desFile)
    model = fasttext.cbow('/data/disk1/private/yx/all.txt',
                              desFile,
                              lr = 0.03,
                              dim = 64 ,
                              silent = 0,
                              thread = 20,
                          min_count = 200,
                                epoch = base
                         )
    (wordnum, vec_size) = (len(model.words), model.dim)
    word2id = {}
    vecList = []
    for idx, word in enumerate(model.words):
        word2id[word] = idx
        vecList.append(model[word])
    with open(destDir+"word2id"+str(base)+".pkl", "wb") as f:
        pickle.dump((wordnum, vec_size), f)
        pickle.dump(word2id, f)
    import numpy as np

    vecnp = np.asarray(vecList)
    print(vecnp.shape)
    np.save(destDir+"vec_nor"+str(base)+".npy", vecnp)
    base = base + base