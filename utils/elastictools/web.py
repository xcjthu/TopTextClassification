from flask import Flask, request, render_template
import logging
import os
import json

from elastic.elastic import search
import elasticsearch.exceptions

app = Flask(__name__)

if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "local_config.py")):
    import local_config as config
else:
    import config as config


@app.route("/")
def root():
    if "query" in request.args:
        print(request.args)
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
        print(data)
        return render_template("main.html", text=json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),query=str(request.args["query"])).replace("\n","<br>")
    else:
        return render_template("main.html")


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)
