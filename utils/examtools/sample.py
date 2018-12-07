import random
import json

d = []
f = open("0_train.json", "r")
for line in f:
    d.append(json.loads(line))


def s():
    i = random.randint(0, len(d) - 1)
    x = ["A", "B", "C", "D"][random.randint(0, 3)]
    print(d[i]["statement"])
    print(d[i]["option_list"][x])
    print(d[i]["reference"][x][0])


while True:
    input()
    s()
