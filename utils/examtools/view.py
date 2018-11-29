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
        if filename.startswith("type"):
            continue
        if not (filename.startswith("xuefa")) or filename.find("related") != -1:
            continue
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


    def dump(d, f):
        f = open(f, "w")
        for x in d:
            print(json.dumps(x, ensure_ascii=False, sort_keys=True), file=f)
        f.close()


    dump(data1[0], os.path.join(path, "type1_train.json"))
    dump(data1[1], os.path.join(path, "type1_test.json"))
    dump(data2[0], os.path.join(path, "type2_train.json"))
    dump(data2[1], os.path.join(path, "type2_test.json"))
