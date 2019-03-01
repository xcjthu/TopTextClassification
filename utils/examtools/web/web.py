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
dara2 = []


def init_question():
    global data
    data = json.load(open("/data/disk3/private/zhx/exam/data/solve/x.json", "r"))
    data2 = json.load(open("/home/zhx/final_biao.json", "r"))


@app.route("/xcj")
def xcj():
    global data
    result = []
    option = ""
    num = ""
    if "option" in request.args.keys() and "num" in request.args.keys():
        option = request.args["option"]
        num = request.args["num"]
        if num == "":
            pass
        else:
            num = int(request.args["num"])
            for a in range(0, 18):
                result.append(data[num]["reference"][option][a])
    for a in range(0, len(result)):
        result[a] = str(a + 1) + ". " + result[a]
    return render_template("main.html", result=result, option=option, num=num)


@app.route("/")
def root():
    global data
    result = []
    option = ""
    num = ""
    if "option" in request.args.keys() and "num" in request.args.keys():
        option = request.args["option"]
        num = request.args["num"]
        if num == "":
            pass
        else:
            num = int(request.args["num"])
            for a in range(0, 18):
                result.append(data[num]["reference"][option][a])
    for a in range(0, len(result)):
        result[a] = str(a + 1) + ". " + result[a]
    return render_template("main.html", result=result, option=option, num=num)


if __name__ == "__main__":
    init_question()
    app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)
