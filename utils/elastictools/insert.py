from elastic.elastic import create_index, delete_index
import json
import os

index_name = "law"
doc_type_map_dic = {
    "国际法": 4,
    "刑法": 2,
    "刑事诉讼法【最新更新】": 2,
    "司法制度和法律职业道德": 1,
    "法制史": 5,
    "民法": 3,
    "民诉与仲裁【最新更新】": 3,
    "国际经济法": 4,
    "法理学": 1,
    "法考冲刺试题": 0,
    "法考真题(按年度)": 0,
    "国际私法": 4,
    "社会主义法治理念": 1,
    "商法": 3,
    "民诉与仲裁【更新中】": 3,
    "行政法与行政诉讼法": 2,
    "宪法": 1,
    "经济法": 4
}

text = ["content"]
keyword = ["type1", "type2", "type3"]
date_type = []

path = "../examtools/data"

if __name__ == "__main__":
    try:
        delete_index(index_name)
    except Exception as e:
        print(e)

    mapping = {}
    for key in text:
        mapping[key] = {
            "type": "text",
            "analyzer": "ik_max_word",
            "search_analyzer": "ik_smart"
        }

    for key in keyword:
        mapping[key] = {
            "type": "keyword"
        }

    for key in date_type:
        mapping[key] = {
            "type": "date"
        }

    mapping = {"mappings": {}}
    for key in doc_type_map_dic.keys():
        mapping["mappings"][key] = {"properties": mapping}

    print(json.dumps(mapping, indent=2))
    create_index(index_name, json.dumps(mapping))

    for type1 in os.listdir(path):
        for type2 in os.listdir(os.path.join(path, type1)):
            print(type2)
            for type3 in os.listdir(os.path.join(path, type1, type2)):
                pass
