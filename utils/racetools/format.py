import os
import json

input_path = "/data/disk3/private/zhx/RACE/data/origin_data"
output_path = "/data/disk3/private/zhx/RACE/data/format_data"

word_set = {}


def add(x):
    x = x.replace("\n", " ")
    x = x.split(" ")
    symbol = [",", ".", "?", "\"", "(", ")", ":", "_", "$", "\'", "!"]
    for y in x:
        for z in symbol:
            y = y.replace(z, "")

        y = y.lower()
        if not (y in word_set.keys()):
            word_set[y] = 0
        word_set[y] += 1


def work(in_path, out_path):
    inf = open(in_path, "r")

    data = json.loads(inf.readline())

    inf.close()

    result = []

    add(data["article"])

    for a in range(0, len(data["answers"])):
        problem = {
            "answer": data["answers"][a],
            "option": data["options"][a],
            "question": data["questions"][a],
            "article": data["article"]
        }
        add(problem["answer"])
        for x in problem["option"]:
            add(x)
        add(problem["question"])
        result.append(problem)

    ouf = open(out_path, "w")
    for data in result:
        print(json.dumps(data, sort_keys=True), file=ouf)
    ouf.close()


def dfs_search(in_path, out_path):
    for filename in os.listdir(in_path):
        if os.path.isdir(os.path.join(in_path, filename)):
            os.makedirs(os.path.join(out_path, filename), exist_ok=True)
            dfs_search(os.path.join(in_path, filename), os.path.join(out_path, filename))
        else:
            work(os.path.join(in_path, filename), os.path.join(out_path, filename))


if __name__ == "__main__":
    dfs_search(input_path, output_path)

    word_x = ["PAD", "UNK"]

    for word in word_set:
        if word_set[word] >= 10:
            word_x.append(word)

    word_dic = {}
    for a in range(0, len(word_x)):
        word_dic[word_x[a]] = a

    json.dump(word_dic, open("/data/disk3/private/zhx/RACE/data/embedding/word2id.txt", "w"), indent=2,
              ensure_ascii=False)
