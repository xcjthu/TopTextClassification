import os
import json
import random

pre_path = "/data/disk3/private/zhx/exam/data/origin_data/学法"
file_list = ["xuefa_data_1.json", "xuefa_data_2.json", "xuefa_data_3.json"]
output_path = "/data/disk3/private/zhx/exam/data/origin_data/format"


def check(d):
    for x in ["甲", "乙", "丙", "丁", "某", "某某"]:
        if d["statement"].find(x) != -1:
            return True
        for option in d["option_list"].keys():
            if d["option_list"][option].find(x) != -1:
                return True
    return False


def dump(data, filename):
    print(filename, len(data))
    f = open(os.path.join(output_path, filename), "w")
    for d in data:
        print(json.dumps(d, ensure_ascii=False), file=f)


map_dic = {
    "国际法": 4,
    "刑法": 2,
    "刑事诉讼法【最新更新】": 2,
    "司法制度和法律职业道德": 1,
    "法制史": 5,
    "民法": 3,
    "民诉与仲裁【最新更新】": 3,
    "国际经济法": 4,
    "法理学": 1,
    "法考冲刺试题": 0,
    "法考真题(按年度)": 0,
    "国际私法": 4,
    "社会主义法治理念": 1,
    "商法": 3,
    "民诉与仲裁【更新中】": 3,
    "行政法与行政诉讼法": 2,
    "宪法": 1,
    "经济法": 4,
}

if __name__ == "__main__":
    data = []
    for a in range(0, 6):
        data.append([[[], []], [[], []]])

    for filename in file_list:
        f = open(os.path.join(pre_path, filename), "r")

        for line in f:
            d = json.loads(line)
            type1 = map_dic[d["subject"]]
            if check(d):
                type2 = 0
            else:
                type2 = 1

            if random.randint(1, 5) == 1:
                type3 = 1
            else:
                type3 = 0

            data[type1][type2][type3].append(d)

    for a in range(0, 6):
        for b in range(0, 2):
            for c in range(0, 2):
                if c == 0:
                    x = "train"
                else:
                    x = "test"

                dump(data[a][b][c], "%d_%d_%s.json" % (a, b, x))
