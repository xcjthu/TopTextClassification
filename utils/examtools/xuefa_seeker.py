import os
import json

pre_path = "/data/disk3/private/zhx/exam/data/cut_data"
file_list = ["xuefa_data_1.json", "xuefa_data_2.json", "xuefa_data_3.json"]

se = set()

map_dic = {
    "国际法": 4,
    "刑法": 2,
    "刑事诉讼法【最新更新】": 2,
    "司法制度和法律职业道德": 1,
    "法制史": -1,
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
    data = [[], [], [], [], []]
    for filename in file_list:
        f = open(os.path.join(pre_path, filename), "r")

        for line in f:
            d = json.loads(line)
            data[map_dic[d["subject"]]].append(d)

    for a in range(0, len(data)):
        f = open(os.path.join(pre_path, "xuefa_split_%d.json" % a), "w")
        for x in data[a]:
            print(json.dumps(x, ensure_ascii=False), file=f)
        f.close()
