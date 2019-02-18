from elastic.elastic import create_index, delete_index, insert_doc

import json
import os
import time
import multiprocessing
import uuid

index_name = "law_laws"
doc_type = "data"

text = ["title", "content"]


def print_time():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def print_info(s):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("[%s] %s" % (times, s), flush=True)


def dfs(x):
    z = x["article_content"]
    for y in x["child"]:
        z = z + " " + dfs(y)
    return z


def dfs_insert(index, doc_type, title, x):
    if x["article_level"] == 4:
        z = dfs(x)
        insert_doc(index, doc_type, {"title": title, "content": z}, str(uuid.uuid4()))
    else:
        for y in x["child"]:
            dfs_insert(index, doc_type, title, y)


def insert_file(index, doc_type, file_path):
    data = json.load(open(file_path, "r"))

    total = len(data["laws"])
    cnt = 0

    for x in data["laws"]:
        for y in x["law_articles"]:
            dfs_insert(index, doc_type, x["title"], y)
        cnt += 1

        print('\r', end='', flush=True)
        print("%d/%d" % (cnt, total), end='', flush=True)

    print("")


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

    mapping = {"mappings": {doc_type: {"properties": mapping}},
               "settings": {
                   "number_of_replicas": 0,
                   "number_of_shards": 30, "analysis": {
                       "analyzer": {
                           "zhx_ngram": {
                               "tokenizer": "zhx_ngram"
                           }
                       },
                       "tokenizer": {
                           "zhx_ngram": {
                               "type": "ngram",
                               "min_gram": 1,
                               "max_gram": 1,
                               "token_chars": [
                                   "letter",
                                   "digit",
                                   "punctuation",
                                   "symbol"
                               ]
                           }
                       }
                   }
               }
               }

    # print(json.dumps(mapping, indent=2))
    create_index(index_name, json.dumps(mapping))

    file_list = ["b0_new.json", "b1_new.json", "b2_new.json", "b3_new.json", "b4_new.json"]
    pre_path = "/data/disk1/data/law_data/"

    for name in file_list:
        name = pre_path + name
        print(name)
        insert_file(index_name, doc_type, name)
