import json
import os

path = "/data/disk3/private/zhx/exam/data/cut_data/format"

file_list = ["0_train.json"]
if __name__ == "__main__":
    cnt = {}
    for filename in file_list:
        f = open(os.path.join(path, filename), "r")
        for line in f:
            d = json.loads(f)
            for option in d["reference"]:
                for r in d["reference"][option]:
                    x = len(r)
                    if not (x in cnt.keys()):
                        cnt[x] = 0
                    cnt[x] += 1

    print(json.dumps(cnt, indent=2, sort_keys=True, ensure_ascii=False))
