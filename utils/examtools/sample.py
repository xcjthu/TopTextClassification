import random
import json

d = []
f = open("/data/disk3/private/zhx/exam/data/origin_data/format/0_train.json", "r")
for line in f:
    d.append(json.loads(line))


def s():
    i = random.randint(0, len(d) - 1)
    x = ["A", "B", "C", "D"][random.randint(0, 3)]
    print(d[i]["subject"], x, d[i]["answer"])
    print(d[i]["statement"])
    print(d[i]["option_list"][x])
    for a in range(0, 10):
        print(a, d[i]["reference"][x][a])


while True:
    input()
    s()
