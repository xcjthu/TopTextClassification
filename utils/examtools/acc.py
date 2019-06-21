import json

data = json.load(open("/home/zhx/ta_all.json", "r", encoding="utf8"))

file_list = [
    "/data/disk3/private/zhx/exam/data/cut_data/test/0.json",
    # "/data/disk3/private/zhx/exam/data/cut_data/test/1.json",
]

map_dic = {
    0: [],
    1: ["A"],
    2: ["B"],
    3: ["A", "B"],
    4: ["C"],
    5: ["A", "C"],
    6: ["B", "C"],
    7: ["A", "B", "C"],
    8: ["D"],
    9: ["A", "D"],
    10: ["B", "D"],
    11: ["A", "B", "D"],
    12: ["C", "D"],
    13: ["A", "C", "D"],
    14: ["B", "C", "D"],
    15: ["A", "B", "C", "D"],
}

correct = {}
cnt = {}

c = 0
for filename in file_list:
    f = open(filename, "r")
    for line in f:
        x = json.loads(line)
        res1 = map_dic[data[c]["res"]]
        res2 = x["answer"]
        co = False
        if set(res1) == set(res2):
            co = True

        for k in x["ans"]:
            if not (k in cnt.keys()):
                cnt[k] = 0
                correct[k] = 0
            if co:
                correct[k] += 1
            cnt[k] += 1

        c += 1

ke = [
    "Word_Match",
    "Concept",
    "Num_analysis",
    "Multi-Paragraph",
    "Multihop-reasoning",
    "命中",
    "部分命中",
    "没有命中",
]
for k in ke:
    print(k, correct[k] / cnt[k])
