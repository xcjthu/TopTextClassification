from elastic.elastic import create_index, delete_index, insert_doc

import json
import os
import time
import multiprocessing
import uuid

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


num_list = {
    u"〇": 0,
    u"\uff2f": 0,
    u"\u3007": 0,
    u"\u25cb": 0,
    u"\uff10": 0,
    u"\u039f": 0,
    u'零': 0,
    "O": 0,
    "0": 0,
    u"一": 1,
    u"元": 1,
    u"1": 1,
    u"二": 2,
    u"2": 2,
    u'三': 3,
    u'3': 3,
    u'四': 4,
    u'4': 4,
    u'五': 5,
    u'5': 5,
    u'六': 6,
    u'6': 6,
    u'七': 7,
    u'7': 7,
    u'八': 8,
    u'8': 8,
    u'九': 9,
    u'9': 9,
    u'十': 10,
    u'百': 100,
    u'千': 1000,
    u'万': 10000
}


def get_number_from_string(s):
    for x in s:
        if not (x in num_list):
            print(s)
            gg

    value = 0
    try:
        value = int(s)
    except ValueError:
        nowbase = 1
        addnew = True
        for a in range(len(s) - 1, -1, -1):
            if s[a] == u'十':
                nowbase = 10
                addnew = False
            elif s[a] == u'百':
                nowbase = 100
                addnew = False
            elif s[a] == u'千':
                nowbase = 1000
                addnew = False
            elif s[a] == u'万':
                nowbase = 10000
                addnew = False
            else:
                value = value + nowbase * num_list[s[a]]
                addnew = True

        if not (addnew):
            value += nowbase

    return value


def parse(s):
    res = []
    temps = ""

    pre = 0

    for a in range(0, len(s)):
        if s[a] == "第":
            b = a + 1
            while b < len(s) and s[b] in num_list.keys():
                b += 1
            if s[b] == "条":
                x = get_number_from_string(s[a + 1:b])
                if x == pre + 1:
                    pre += 1
                    if len(temps) != 0:
                        res.append(temps)
                    temps = s[a:b + 1]
                    a = b
                else:
                    temps = temps + s[a]

        else:
            temps = temps + s[a]

    if len(temps) != 0:
        res.append(temps)

    return res


def insert_file(index, doc_type, file_path):
    inf = open(file_path, "r")

    for line in inf:
        try:
            data = json.loads(line)
            uid = data["uniqid"]
            data.pop("uniqid")
            data["issue_date"] = data["issue_date"].split(" ")[0]

            content = parse(data["content"])
            print(json.dumps(content, indent=2, ensure_ascii=False))
            break
            for x in content:
                insert_doc(index, doc_type, data, str(uuid.uuid4()))
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
