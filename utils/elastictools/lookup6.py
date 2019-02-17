import json
import os

from elastic.elastic import search

# path = "/data/disk3/private/zhx/exam/data/origin_data/format/"
path = "/data/disk3/private/zhx/exam/data/origin_data/final2"
output_path = "/data/disk3/private/zhx/exam/data/origin_data/final3"
file_list = ["0_train.json", "1_train.json", "0_test.json", "1_test.json"]


def workx(s, t, k):
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
    if not (t is None):
        request_body["query"]["bool"]["must"] = [
            {
                "term": {
                    "type2": {
                        "value": t
                    }
                }
            }
        ]

    response = search("law", "data", request_body)
    for a in range(0, k):
        l.append(response["hits"]["hits"][a]["_source"]["content"])

    return l


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

    return l


def work(filename):
    print(filename)
    data = []
    f = open(os.path.join(path, filename), "r")
    for line in f:
        d = json.loads(line)
        d["reference"] = {}
        for option in d["option_list"]:
            query = d["statement"] + " " + d["option_list"][option]
            d["reference"][option] = workx(query, d["subject"][0], 6) + workx(query, d["subject"][1], 6) + worky(query,
                                                                                                                 6)

        data.append(d)

    f.close()
    f = open(os.path.join(output_path, filename), "w")
    for d in data:
        print(json.dumps(d, sort_keys=True, ensure_ascii=False), file=f)
    f.close()


if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    for filename in file_list:
        work(filename)
