import json
import os

from elastic.elastic import search

# path = "/data/disk3/private/zhx/exam/data/origin_data/format/"
path = "/data/disk3/private/zhx/exam/data/origin_data/final3"
output_path = "/data/disk3/private/zhx/exam/data/origin_data/final4"
file_list = ["0_train.json", "1_train.json", "0_test.json", "1_test.json"]


def format(s):
    s = s.replace(" ", "").replace("\t", "").replace("\n", "")
    return s


def worky(s, k):
    l = []
    request_body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "content": s
                        }
                    }
                ]
            }
        }
    }
    response = search("law_laws", "data", request_body)
    for a in range(0, k):
        l.append(response["hits"]["hits"][a]["_source"]["content"])
    # print(json.dumps(l, indent=2, ensure_ascii=False))

    return l


def work(filename):
    total = 0
    f = open(os.path.join(path, filename), "r")
    for line in f:
        total += 1
    f.close()

    print(filename)
    data = []
    cnt = 0
    f = open(os.path.join(path, filename), "r")
    for line in f:
        d = json.loads(line)
        for option in d["option_list"]:
            query = d["statement"] + " " + d["option_list"][option]
            d["reference"][option] = d["reference"][option][0:12] + worky(query, 6)

        data.append(d)
        cnt += 1
        print("\r", end='')
        print("%d/%d" % (cnt, total), end='')

    f.close()
    f = open(os.path.join(output_path, filename), "w")
    for d in data:
        print(json.dumps(d, sort_keys=True, ensure_ascii=False), file=f)
    f.close()


if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    for filename in file_list:
        work(filename)
