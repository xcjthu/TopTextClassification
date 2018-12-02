import jieba
import os
import json

input_data_path = "/data/disk3/private/zhx/exam/data/origin_data"
output_data_path = "/data/disk3/private/zhx/exam/data/cut_data"

word_set = set()


def cut(content):
    content = list(jieba.cut(content))
    for word in content:
        word_set.add(word)
    return content


def cut_file(path):
    os.makedirs(os.path.join(output_data_path, "/".join(path.split("/")[:-1])), exist_ok=True)
    input_file = open(os.path.join(input_data_path, path), "r")
    output_file = open(os.path.join(output_data_path, path), "w")

    for line in input_file:
        try:
            data = json.loads(line)
            data["statement"] = cut(data["statement"])

            for option in data["option_list"]:
                data["option_list"][option] = cut(data["option_list"][option])

            if "analyse" in data.keys():
                data["analyse"] = cut(data["analyse"])

            if "reference" in data.keys:
                for option in data["reference"]:
                    for a in range(0, len(data["reference"][option])):
                        data["reference"][option][a] = cut(data["reference"][option][a])

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

json.dump(word_dic, open("/data/disk3/private/zhx/exam/data/embedding/word2id.txt", "w"), indent=2,
          ensure_ascii=False)
