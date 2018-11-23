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


for filename in os.listdir(input_data_path):
    input_file = open(os.path.join(input_data_path, filename), "r")
    output_file = open(os.path.join(output_data_path, filename), "w")

    for line in input_file:
        try:
            data = json.loads(line)
            data["statement"] = cut(data["statement"])

            if not ("option_list" in data.keys()):
                data["option_list"] = data["option"]
                data.pop("option")

            for option in data["option_list"]:
                data["option_list"][option] = cut(data["option_list"][option])

            print(json.dumps(data, ensure_ascii=False), file=output_file)

        except Exception as e:
            pass

    input_file.close()
    output_file.close()

word_set = ["PAD", "UNK"] + list(word_set)
word_dic = {}
for a in range(0, len(word_set)):
    word_dic[word_set[a]] = a

json.dump(word_dic, open("/data/disk3/private/zhx/exam/data/embedding/word2id.txt", "w"), indent=2,
          ensure_ascii=False)
