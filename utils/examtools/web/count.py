import argparse
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f')
args = parser.parse_args()

temp = json.load(open(args.file, "r", encoding="utf8"))

cnt = 0
dic = {}
for x in temp["term"]:
    # if x["userName"] != "xiaocj":
    #    continue
    # print(x["content"][0]["content"][0])
    match = re.search(r"num=(\d+)", x["content"][0]["content"][0])
    idx = int(match.group(1))
    cnt += 1
    for y in x["result"]:
        for z in y["answer"]:
            if z not in dic.keys():
                dic[z] = 0
            dic[z] += 1

print(cnt)
print(json.dumps(dic, indent=2, ensure_ascii=False, sort_keys=True))
