from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import json
import re

from elastic.elastic import search

app = Flask(__name__, static_folder=os.path.join(os.getcwd(), "static"),
            static_url_path='/static',
            template_folder=os.path.join(os.getcwd(), "templates"))

if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "local_config.py")):
    import local_config as config
else:
    import config as config

res = {}


def parse_data(data, father_id, father):
    for x in data:
        if type(x) is str:
            if not (father_id in res.keys()):
                res[father_id] = []
                for y in father:
                    res[father_id].append(y)
                res[father_id].append(father_id)
            res[father_id].append(x)

        else:
            parse_data(x["content"], x["_id"], father + [x["_name"]])


@app.route("/get")
def get():
    if "id" in request.args:
        return render_template("get.html", result=res[request.args["id"]])

    return ""


@app.route("/check")
def check():
    id_ = request.args["id"]
    if id_ in res.keys():
        return redirect("/get?id=%s" % id_)
    else:
        return redirect("/view?id=%s" % id_)


@app.route("/search")
def search_():
    text = ""
    result = []
    which = 1
    if "text" in request.args:
        body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "text": request.args["text"]
                            }
                        }
                    ]
                }
            }
        }
        text = request.args["text"]
        which = int(request.args["which"])
        if which == 1:
            index = "content"
        else:
            index = "content2"

        data = search(index, "data", body, size=20)
        for x in data["hits"]["hits"]:
            if which == 1:
                name = x["_source"]["text"]
                id_ = x["_source"]["id"]
            else:
                name = x["_source"]["name"]
                id_ = x["_source"]["id"]
            if id_ in res.keys():
                result.append({"name": name, "id": id_, "xcj": "最底层"})
            else:
                result.append({"name": name, "id": id_, "xcj": ""})
    return render_template("search.html", result=result, text=text, which=which)


def prefix(id_, x):
    a = id_.split("-")
    b = x.split("-")
    if len(a) < len(b):
        return False
    for c in range(0, len(b)):
        if b[c] != a[c]:
            return False
    return True


@app.route("/view")
def view():
    s = render_template("view.html")
    if "id" in request.args:
        id_ = request.args["id"]
        arr = s.split("\n")
        for a in range(0, len(arr)):
            s = arr[a]

            if s.find("window.location") != -1:
                arr[a] = arr[a].replace("#home", "#home-" + id_.replace("-", "-collapse-") + "-collapse")
            elif s.find("data-toggle") != -1 and s.find("aria-expanded") != -1:
                pattern = re.compile("data-target=\"#(\S*)\"")
                result = pattern.findall(s)
                x = str(result[0]).replace("home-", "").replace("-collapse", "")
                if prefix(id_, x):
                    arr[a] = arr[a].replace("false", "true").replace("btn-link collapsed", "btn-link")
            elif s.find("data-parent") != -1:
                pattern = re.compile("id=\"(\S*)\"")
                result = pattern.findall(s)
                x = str(result[0]).replace("home-", "").replace("-collapse", "")
                if prefix(id_, x):
                    arr[a] = arr[a].replace("class=\"collapse\"", "class=\"collapse show\"")

        s = "\n".join(arr)

    return s


@app.route("/note")
def note():
    return send_from_directory("", "x.pdf")


if __name__ == "__main__":
    parse_data(json.load(open("data2.json", "r", encoding="utf8")), "", [])
    app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)
