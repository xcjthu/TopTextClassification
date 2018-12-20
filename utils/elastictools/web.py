from flask import Flask, request, render_template
import logging
import os
import json
import jieba
import nltk

from elastic.elastic import search
import elasticsearch.exceptions

app = Flask(__name__)

if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "local_config.py")):
    import local_config as config
else:
    import config as config

problem_collection = ""
kaodian_collection = ""


def init_problem():
    data = []

    path = "/data/disk3/private/zhx/exam/data/origin_data/gen/4"
    for filename in os.listdir(path):
        if filename.startswith("1"):
            continue
        f = open(os.path.join(path, filename), "r")
        for line in f:
            x = json.loads(line)
            for option in ["A", "B", "C", "D"]:
                data.append(x["statement"] + " " + x["option_list"][option])

    problem_data = []
    for x in data:
        problem_data.append(list(jieba.cut(x)))

    global problem_collection
    problem_collection = nltk.text.TextCollection(problem_data)


datax = []


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

    datax.append(temp)


def init_kaodian():
    path = "../examtools/data"
    for type1 in os.listdir(path):
        for type2 in os.listdir(os.path.join(path, type1)):
            print(type2)
            for type3 in os.listdir(os.path.join(path, type1, type2)):
                data = json.load(open(os.path.join(path, type1, type2, type3), "r"))
                dfs_insert(type1, type2, type3, data)

    kaodian_data = []
    for x in datax:
        kaodian_data.append(list(jieba.cut(x)))

    global kaodian_collection
    kaodian_collection = nltk.text.TextCollection(kaodian_data)


def get(data, collection):
    data = list(jieba.cut(data))
    arr = []
    for word in data:
        arr.append(word)
    arr = list(set(arr))

    for b in range(0, len(arr)):
        arr[b] = (collection.tf_idf(arr[b], data), arr[b])

    arr.sort(reverse=True)

    arr = arr[:10]

    return arr


@app.route("/")
def root():
    if "query" in request.args:
        data = search("law", "data", {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "content": request.args["query"]
                            }
                        }
                    ],
                }
            }
        })["hits"]["hits"]
        problem = get(request.args["query"], problem_collection)
        kaodian = get(request.args["query"], kaodian_collection)
        return render_template("main.html", text=json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
                               problem=json.dumps(problem, ensure_ascii=False),
                               kaodian=json.dumps(kaodian, ensure_ascii=False),
                               query=str(request.args["query"])).replace("\n", "<br>")
    else:
        return render_template("main.html").replace("\n", "<br>")


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)
