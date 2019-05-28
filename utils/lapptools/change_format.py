import json
import random


def shuffle_train(filepath, outpath):
    fin = open(filepath, 'r')
    fout = open(outpath, 'w')

    for d in fin:
        d = json.loads(d)
        if random.randint(0, 1) == 0:
            d['label'] = 'B'
        else:
            d['label'] = 'C'
            B = d['B']
            C = d['C']
            d['B'] = C
            d['C'] = B
        print(json.dumps(d, ensure_ascii = False), file = fout)


def shuffle_test(filein, filelabel, outpath):
    fin = open(filein, 'r').readlines()
    filelabel = open(filelabel, 'r').readlines()

    fout = open(outpath, 'w')
    
    for i in range(len(fin)):
        d = json.loads(fin[i])
        d['label'] = filelabel[i].strip()
        print(json.dumps(d, ensure_ascii = False), file = fout)


if __name__ == '__main__':
    shuffle_train('/data/disk1/private/xcj/LAPP/data/cut_data/cail_small.txt', '/data/disk1/private/xcj/LAPP/data/cut_data/cail_small_shuffle.txt')
    shuffle_test('/data/disk1/private/xcj/LAPP/data/cut_data/valid_small.txt', '/data/disk1/private/zhx/cail2019/la/small/output/output.txt', '/data/disk1/private/xcj/LAPP/data/cut_data/valid_small_shuffle.txt')

