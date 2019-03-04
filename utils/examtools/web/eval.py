import argparse
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f')
args = parser.parse_args()

answer = json.load(open("res/a.json", "r"))
temp = json.load(open(args.file, "r", encoding="utf8"))

res = []
for x in temp["term"]:
    # if x["userName"] != "louky":
    #    continue
    # print(x["content"][0]["content"][0])
    match = re.search(r"num=(\d+)", x["content"][0]["content"][0])
    idx = int(match.group(1))
    for y in x["result"]:
        res.append({"id": idx, "res": y["answer"]})

dic = {}
for a in range(0, 4):
    dic[a] = {"correct": 0, "total": 0}

for x in res:
    idx = x["id"]
    t = answer[idx]["type"]
    dic[t]["total"] += 1
    if set(answer[idx]["ans"]) == set(x["res"]):
        dic[t]["correct"] += 1

# print(json.dumps(dic, indent=2, ensure_ascii=False, sort_keys=True))

print(
    dic[0]["correct"] / dic[0]["total"] * 100,
    (dic[0]["correct"] + dic[1]["correct"]) / (dic[0]["total"] + dic[1]["total"]) * 100,
    dic[2]["correct"] / dic[2]["total"] * 100,
    (dic[2]["correct"] + dic[3]["correct"]) / (dic[2]["total"] + dic[3]["total"]) * 100,
    (dic[0]["correct"] + dic[2]["correct"]) / (dic[0]["total"] + dic[2]["total"]) * 100,
    (dic[0]["correct"] + dic[1]["correct"] + dic[2]["correct"] + dic[3]["correct"]) / (
            dic[0]["total"] + dic[1]["total"] + dic[2]["total"] + dic[3]["total"]) * 100,
    sep="\t"
)
