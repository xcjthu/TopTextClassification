import thulac
import json
import os


inpath = '/data/disk1/private/xcj/LAPP/data/origin_data'
outpath = '/data/disk1/private/xcj/LAPP/data/cut_data'

filelist = ['cail_small.txt', 'valid_small.txt']

wordidpath = '/data/disk1/private/xcj/LAPP/data/word2id.txt'

cutter = thulac.thulac(seg_only = True)

def cut(text):
    text = text.replace('\n', '')
    ans = cutter.cut(text)
    return [v[0] for v in ans]

word_list = {}

def cut_file(filename):
    fin = open(os.path.join(inpath, filename), 'r')
    fout = open(os.path.join(outpath, filename), 'w')
    
    global word_list

    for d in fin:
        d = json.loads(d)
        tmp = {}
        for key in d.keys():
            tmp[key] = cut(d[key])
            for w in tmp[key]:
                if w in word_list:
                    word_list[w] += 1
                else:
                    word_list[w] = 1

        print(json.dumps(tmp, ensure_ascii = False), file = fout)
    


def cut_data():
    global filelist
    
    for f in filelist:
        cut_file(f)


if __name__ == '__main__':
    cut_data()
    fout = open(wordidpath, 'w')

    word2id = {}
    for key in word_list.keys():
        if word_list[key] <= 5:
            continue
        word2id[key] = len(word2id)
    
    word2id['UNK'] = len(word2id)
    word2id['PAD'] = len(word2id)

    print(json.dumps(word2id, ensure_ascii = False), file = fout)
    fout.close()


