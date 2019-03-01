import jieba
import os
import json
import requests
from requests.auth import HTTPBasicAuth

input_data_path = "/data/disk1/private/zhx/ffzj/origin_data"
output_data_path = "//data/disk1/private/zhx/ffzj/cut_data2"

word_set = set()

username = None
password = None


def cut(content):
    data = list(jieba.cut(content))
    content = []
    for x in data:
        word = x
        content.append(word)
        word_set.add(word)

    return content


def cut_file(path):
    print(path)
    os.makedirs(os.path.join(output_data_path, "/".join(path.split("/")[:-1])), exist_ok=True)
    input_file = open(os.path.join(input_data_path, path), "r")
    output_file = open(os.path.join(output_data_path, path), "w")

    for line in input_file:
        try:
            data = line#json.loads(line)
            data = cut(data.replace("\n",""))#data["text"] = cut(data["text"].replace("\n", ""))

            print(json.dumps(data, ensure_ascii=False, sort_keys=True), file=output_file)

        except Exception as e:
            print(e)
            raise e

    input_file.close()
    output_file.close()


def dfs_search(path):
    real_path = os.path.join(input_data_path, path)
    for filename in os.listdir(real_path):
        file_path = os.path.join(real_path, filename)
        if os.path.isdir(file_path):
            dfs_search(os.path.join(path, filename))
        else:
            cut_file(os.path.join(path, filename))


dfs_search("")

word_set = ["PAD", "UNK"] + list(word_set)
word_dic = {}
for a in range(0, len(word_set)):
    word_dic[word_set[a]] = a

json.dump(word_dic, open("/data/disk1/private/zhx/ffzj/word2id.txt", "w"), indent=2,
          ensure_ascii=False)
