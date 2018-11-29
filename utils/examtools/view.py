import json
import os
import random

path = "/data/disk3/private/zhx/exam/data/origin_data"


def check(s):
    for x in ["甲", "乙", "丙", "丁", "某", "某某"]:
        if s.find(x) != -1:
            return True
    return False


if __name__ == "__main__":
    cnt1 = 0
    cnt2 = 0
    data1 = [[], []]
    data2 = [[], []]
    for filename in os.listdir(path):
        f = open(os.path.join(path, filename), "r")
        op = random.randint(1, 5)
        if op == 1:
            op = 1
        else:
            op = 0
        for line in f:
            data = json.loads(line)
            if check(data["statement"]):
                cnt1 += 1
                data1[op].append(data)
            else:
                cnt2 += 1
                data2[op].append(data)

    print(cnt1, cnt2)
    json.dump(data1[0], open(os.path.join(path, "type1_train.json"), "w"), ensure_ascii=False)
    json.dump(data1[1], open(os.path.join(path, "type1_test.json"), "w"), ensure_ascii=False)
    json.dump(data2[0], open(os.path.join(path, "type1_train.json"), "w"), ensure_ascii=False)
    json.dump(data2[1], open(os.path.join(path, "type1_test.json"), "w"), ensure_ascii=False)
