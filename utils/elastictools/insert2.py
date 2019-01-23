from elastic.elastic import create_index, delete_index, insert_doc
import json
import os

index_name = "law2"
doc_type = "data"

unknown_list = ["知识产权法",
                "劳动与社会保障法",
                "环境资源法"]

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

text = ["content"]
keyword = ["type1", "type2", "type3"]
date_type = []

path = "../examtools/data2"
cnt = 0


def dfs_insert(type1, type2, type3, data):
    global cnt
    cnt += 1
    tx = cnt
    temp = ""
    for x in data:
        if type(x) is list:
            dfs_insert(type1, type2, type3, x)
        else:
            temp = temp + x + " "

    insert_doc(index_name, doc_type, {"content": temp, "type1": type1, "type2": type2, "type3": type3}, tx)


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

    mapping = {"mappings": {doc_type: {"properties": mapping}}}

    print(json.dumps(mapping, indent=2))
    create_index(index_name, json.dumps(mapping))

    for type1 in os.listdir(path):
        for type2 in os.listdir(os.path.join(path, type1)):
            print(type2)
            for type3 in os.listdir(os.path.join(path, type1, type2)):
                data = json.load(open(os.path.join(path, type1, type2, type3), "r"))
                dfs_insert(type1, type2, type3, data)
