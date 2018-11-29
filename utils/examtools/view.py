import json
import os

path = "/data/disk3/private/zhx/exam/data/origin_data"


def check(s):
    for x in ["甲", "乙", "丙", "丁", "某", "某某"]:
        if s.find(x) != -1:
            return True
    return False


if __name__ == "__main__":
    cnt1 = 0
    cnt2 = 0
    for filename in os.listdir(path):
        f = open(os.path.join(path, filename), "r")
        for line in f:
            data = json.loads(line)
            if check(data["statement"]):
                cnt1 += 1
            else:
                cnt2 += 1

    print(cnt1, cnt2)
