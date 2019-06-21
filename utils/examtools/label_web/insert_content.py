import json
import uuid

from elastic.elastic import insert_doc, delete_index, create_index

data = json.load(open("data2.json", "r", encoding="utf8"))

index_name = "content2"
doc_type = "data"


def dfs_search(data, father_name, father_id):
    temp = ""
    insert = False
    for x in data:
        if type(x) is str:
            temp = temp + x + " "
            insert = True
        else:
            dfs_search(x["content"], x["_name"], x["_id"])
    if insert:
        insert_doc(index_name, doc_type, {"text": temp, "id": father_id, "name": father_name}, str(uuid.uuid4()))


if __name__ == "__main__":
    try:
        delete_index(index_name)
    except Exception as e:
        print(e)

    mapping = {}
    for key in ["text"]:
        mapping[key] = {
            "type": "text",
            "analyzer": "ik_max_word",
            "search_analyzer": "ik_smart"
        }

    mapping = {"mappings": {doc_type: {"properties": mapping}},
               "settings": {"number_of_replicas": 0, "number_of_shards": 30}}
    create_index(index_name, json.dumps(mapping))
    cnt = 0
    dfs_search(data, "", "")
