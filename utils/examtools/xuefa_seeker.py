import os
import json

pre_path = "/data/disk3/private/zhx/exam/data/cut_data"
file_list = ["xuefa_data_1.json", "xuefa_data_2.json", "xuefa_data_3.json"]

se = set()

if __name__ == "__main__":
    for filename in file_list:
        f = open(os.path.join(pre_path, filename), "r")

        for line in f:
            d = json.loads(line)
            se.add(d["subject"])

    print(len(se))
    print(json.dumps(list(se), indent=2, ensure_ascii=False))
