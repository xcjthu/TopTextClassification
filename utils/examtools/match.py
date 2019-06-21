import os
import json
import re

data1 = json.load(open("/home/zhx/final_biao.json", "r"))
data2 = json.load(open("/home/zhx/q7.json", "r"))

for a in range(0, len(data1)):
    data1[a].pop("reference")
    data1[a].pop("analyse")
    data1[a].pop("chapter")
    data1[a].pop("subject")
    data1[a].pop("id")
    if data1[a]["type"] == 1:
        data1[a]["rank"] += 1912

data2_res = []
for x in data2["term"]:
    # if x["userName"] != "xiaocj":
    #    continue
    # print(x["content"][0]["content"][0])
    match = re.search(r"num=(\d+)", x["content"][0]["content"][0])
    idx = int(match.group(1))
    for y in x["result"]:
        data2_res.append({"id": idx, "ans": y["answer"]})

print(data2_res[0])

reasoning_keywords = [
    "Word_Match",
    "Concept",
    "Num_analysis",
    "Multi-Paragraph",
    "Multihop-reasoning"
]
missing_keywords = {
    "命中": 0,
    "部分命中": 1,
    "没有命中": 2,
}

reasoning_count = [{}, {}, {}]
for a in range(0, 3):
    for x in reasoning_keywords:
        reasoning_count[a][x] = 0

total_reasoning = [0, 0, 0]
for a in range(0, 3):
    for x in reasoning_keywords:
        total_reasoning[a] += reasoning_count[a][x]
for x in data2_res:
    idx = x["id"]
    t = data1[idx]["type"]

    find = 0
    for y in x["ans"]:
        if y in set(reasoning_keywords):
            reasoning_count[t][y] += 1
            reasoning_count[2][y] += 1
    total_reasoning[2] += 1
    total_reasoning[t] += 1

for x in reasoning_keywords:
    print(x, end='\t')
    print("%.2f" % (reasoning_count[0][x] / total_reasoning[0] * 100), end='%\t')
    print("%.2f" % (reasoning_count[1][x] / total_reasoning[1] * 100), end='%\t')
    print("%.2f" % (reasoning_count[2][x] / total_reasoning[2] * 100), end='%\n')

print("")

missing_count = [{}, {}, {}]
for a in range(0, 3):
    for x in reasoning_keywords:
        missing_count[a][x] = 0
    missing_count[a][0] = 0
    missing_count[a][1] = 0

for x in data2_res:
    idx = x["id"]
    t = data1[idx]["type"]

    for y in x["ans"]:
        if y in set(reasoning_keywords):
            for z in x["ans"]:
                if z in set(missing_keywords):
                    w = missing_keywords[z]

                    missing_count[w][y] += 1
                    missing_count[w][t] += 1

for a in range(0, 3):
    missing_count[a][2] = missing_count[a][0] + missing_count[a][1]

for x in [2, 0, 1]:
    t = 0
    for a in range(0, 3):
        t += missing_count[a][x]
    print(x, end='\t')
    print("%.2f" % (missing_count[0][x] / t * 100), end='%\t')
    print("%.2f" % (missing_count[1][x] / t * 100), end='%\t')
    print("%.2f" % (missing_count[2][x] / t * 100), end='%\n')

for x in reasoning_keywords:
    t = 0
    for a in range(0, 3):
        t += missing_count[a][x]
    print(x, end='\t')
    print("%.2f" % (missing_count[0][x] / t * 100), end='%\t')
    print("%.2f" % (missing_count[1][x] / t * 100), end='%\t')
    print("%.2f" % (missing_count[2][x] / t * 100), end='%\n')

import requests
from requests.auth import HTTPBasicAuth

username = "elastic"
password = "zhx123qazelastic"


def cut(content):
    global username, password
    if password is None:
        print("Enter username: ", end='')
        username = input().replace("\n", "")
        print("Enter password: ", end='')
        password = input().replace("\n", "")

    url = "http://103.242.175.80:9200/_analyze"
    response = requests.get(url, data=json.dumps({"analyzer": "ik_smart", "text": content}),
                            auth=HTTPBasicAuth(username, password),
                            headers={"Content-Type": "application/json"})

    data = json.loads(response.text)["tokens"]
    content = []
    for x in data:
        word = x["token"]
        content.append(word)

    return content


for x in data1:
    x["statement"] = cut(x["statement"])
    for o in ["A", "B", "C", "D"]:
        x["option_list"][o] = cut(x["option_list"][o])


def fo(s):
    return "".join(s).replace("\n", "")


def check(y, x):
    able = True
    if fo(y["statement"]) != fo(x["statement"]):
        return False
    for option in ["A", "B", "C", "D"]:
        if fo(y["option_list"][option]) != fo(x["option_list"][option]):
            return False

    return True


output_file = [
    open("/data/disk3/private/zhx/exam/data/cut_data/test/0.json", "w", encoding="utf8"),
    open("/data/disk3/private/zhx/exam/data/cut_data/test/1.json", "w", encoding="utf8")
]

for x in data2_res:
    if data1[x["id"]]["type"] == -11:
        continue

    # print(x["id"])
    w = data1[x["id"]]
    t = w["type"]
    # print(x)
    # print(w)

    f = open("/data/disk3/private/zhx/exam/data/cut_data/final4/%d_test.json" % t, "r", encoding="utf8")
    find = False
    for line in f:
        y = json.loads(line)
        if check(y, w):
            y["ans"] = x["ans"]
            print(json.dumps(y, ensure_ascii=False, sort_keys=True), file=output_file[t])
            find = True
            break

    if not (find):
        print(w)
        print(x)
        gg

    # break
