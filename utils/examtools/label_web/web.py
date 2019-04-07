from flask import Flask, request, render_template
import os
import json

from elastic.elastic import search

app = Flask(__name__, static_folder=os.path.join(os.getcwd(), "static"),
            static_url_path='/static',
            template_folder=os.path.join(os.getcwd(), "templates"))

if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "local_config.py")):
    import local_config as config
else:
    import config as config


@app.route("/search")
def search_():
    text = ""
    result = []
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

        data = search("content", "data", body, size=20)
        for x in data["hits"]["hits"]:
            result.append({"name": x["_source"]["text"], "id": x["_source"]["id"]})
    return render_template("search.html", result=result, text=text)


@app.route("/view")
def voew():
    return render_template("view.html")


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)
