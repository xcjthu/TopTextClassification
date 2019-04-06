import json

from elastic.elastic import insert_doc, delete_index, create_index

data = json.load(open("data.json", "r", encoding="utf8"))

index_name = "content"
doc_type = "data"


def dfs_search(data):
    cnt = 0
    for x in data:
        insert_doc(index_name, doc_type, {"text": x["_name"], "id": x["_id"]})
        dfs_search(x["content"])


if __name__ == "__main__":
    try:
        delete_index(index_name)
    except Exception as e:
        print(e)

    mapping = {}
    for key in ["text"]:
        mapping[key] = {
            "type": "text",
            "analyzer": "pl_ik_max_word",
            "search_analyzer": "pl_ik_smart"
        }

    mapping = {"mappings": {doc_type: {"properties": mapping}},
               "settings": {"number_of_replicas": 0, "number_of_shards": 30}}
    create_index(index_name, json.dumps(mapping))
    cnt = 0
    dfs_search(data)
