from flask import Flask, request, render_template
import os
import json

app = Flask(__name__, static_folder=os.path.join(os.getcwd(), "static"),
            static_url_path='/static',
            template_folder=os.path.join(os.getcwd(), "templates"))

if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "local_config.py")):
    import local_config as config
else:
    import config as config

data = []


def init_question():
    global data
    data = json.load(open("/data/disk3/private/zhx/exam/data/solve/x.json", "r"))


@app.route("/")
def root():
    result = []
    option = ""
    num = ""
    if "option" in request.args and "num" in request.args:
        option = request.args["option"]
        num = int(request.args["num"])
        result = data[num]["reference"][option]
    for a in range(0, len(result)):
        result[a] = str(a) + ". " + result[a]
    return render_template("main.html", result=result, option=option, num=num)


if __name__ == "__main__":
    init_question()
    app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)
