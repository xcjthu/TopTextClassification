import random
import json

d = []
f = open("/data/disk3/private/zhx/exam/data/origin_data/gen/4/0_train.json", "r")
for line in f:
    d.append(json.loads(line))


def s():
    i = random.randint(0, len(d) - 1)
    x = ["A", "B", "C", "D"][random.randint(0, 3)]
    print(d[i]["subject"], x, d[i]["answer"])
    print(d[i]["analyse"].replace("\n", ""))
    print(d[i]["statement"].replace("\n", ""))
    for x in ["A", "B", "C", "D"]:
        print(x, d[i]["option_list"][x])
    # print(d[i]["option_list"][x].replace("\n", ""))

    # for a in range(0, 10):
    #    print(a, d[i]["reference"][x][a])


while True:
    input()
    s()
