import json
import os

from elastic.elastic import search

#path = "/data/disk3/private/zhx/exam/data/origin_data/format/"
path = "/data/disk1/private/xcj/exam/data/origin_data/"
output_path = "/data/disk1/private/xcj/exam/data/gen/2"
file_list = ["0_train.json", "1_train.json", "0_test.json", "1_test.json"]

doc_type_map_dic = {
    "国际法": "国际法",
    "刑法": "刑法",
    "刑事诉讼法【最新更新】": "刑事诉讼法",
    "司法制度和法律职业道德": "司法制度和法律职业道德",
    "法制史": "目录和中国法律史",
    "民法": "民法",
    "民诉与仲裁【最新更新】": "民事诉讼法",
    "国际经济法": "国际经济法",
    "法理学": "法理学",
    "法考冲刺试题": -1,
    "法考真题(按年度)": -1,
    "国际私法": "国际私法",
    "社会主义法治理念": "中国特色社会主义法治理论",
    "商法": "商法",
    "民诉与仲裁【更新中】": "民事诉讼法",
    "行政法与行政诉讼法": "行政法与行政诉讼法",
    "宪法": "宪法",
    "经济法": "经济法"
}


def work(filename):
    print(filename)
    data = []
    f = open(os.path.join(path, filename), "r")
    for line in f:
        d = json.loads(line)
        if doc_type_map_dic[d["subject"]] != -1:
            type2 = doc_type_map_dic[d["subject"]]

            d["reference"] = {}
            for option in d["option_list"]:
                s1 = d["analyse"]
                s2 = d["option_list"][option]
                request_body = {
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "content": s1
                                    }
                                }
                            ],
                            "must": [
                                {
                                    "term": {
                                        "type2": {
                                            "value": type2
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }

                response = search("law", "data", request_body)
                d["reference"][option] = []
                for a in range(0, 10):
                    d["reference"][option].append(response["hits"]["hits"][a]["_source"]["content"])

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
