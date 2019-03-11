import os
import json
import random

path = "/data/disk1/private/xcj/exam/data/origin_data/"
work_list = ["train", "test"]

if __name__ == "__main__":
    for work in work_list:
        for b in range(0, 2):
            data = []
            for a in range(1, 5):
                f = open(os.path.join(path, "%d_%d_%s.json" % (a, b, work)), "r")
                for line in f:
                    data.append(json.loads(line))

            random.shuffle(data)

            f = open(os.path.join(path, "%d_%s.json" % (b, work)), "w")
            for d in data:
                print(json.dumps(d, ensure_ascii=False, sort_keys=True), file=f)
