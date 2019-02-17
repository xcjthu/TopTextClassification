from elastic.elastic import create_index, delete_index, insert_doc

import json
import os
import time
import multiprocessing

index_name = "law_laws"
doc_type = "data"

text = ["title", "content"]
keyword = ["nid", "issue_department", "authority_level", "validity_status", "document_number"]
date_type = ["issue_date"]


def print_time():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def print_info(s):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("[%s] %s" % (times, s), flush=True)


def insert_file(index, doc_type, file_path):
    inf = open(file_path, "r")

    for line in inf:
        try:
            data = json.loads(line)
            uid = data["uniqid"]
            data.pop("uniqid")
            data["issue_date"] = data["issue_date"].split(" ")[0]

            insert_doc(index, doc_type, data, uid)
        except Exception as e:
            raise e


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

    print(json.dumps(mapping, indent=2))
    create_index(index_name, json.dumps(mapping))

    insert_file(index_name, doc_type, "/home/zhx/laws_es.json")
