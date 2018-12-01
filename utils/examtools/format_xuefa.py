import json
import os

pre_path = "/data/disk3/private/zhx/exam/data/origin_data/学法"

if __name__ == "__main__":
    for filename in os.listdir(pre_path):
        f = open(os.path.join(pre_path, filename), "r")
        data = []
        for line in f:
            d = json.loads(line)
            d["option_list"] = d["option"]
            d.pop("option_list")
            data.append(d)
        f.close()
        f = open(os.path.join(pre_path, filename), "w")
        for d in data:
            print(json.dumps(d, ensure_ascii=False), file=f)
        f.close()
