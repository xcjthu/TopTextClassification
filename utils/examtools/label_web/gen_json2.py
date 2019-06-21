import json
import os

data_path = "/data/disk3/private/zhx/exam/data/origin_data/final4"


def generate_question(source, idx, question, label=None):
    res = {}
    answer = ""
    for option in ["A", "B", "C", "D"]:
        if option in question["answer"]:
            answer += option
    if "analyse" in question.keys():
        analyze = "解析：" + question["analyse"] + "\n"
    else:
        analyze = ""
    description = """
<font size="4">
<div class="text-wrapper" style="background: none;">
    题目来源：%s 题目编号：%d <a href=\"http://103.242.175.80:16018/search\" target=\"_blank\" style=\"color:#55A2F3\">跳转至搜索网站</a>
</div>
%s <br>
选项A：%s <br>
选项B：%s <br>
选项C：%s <br>
选项D：%s <br>
<button type="button" class="el-button operator-item el-button--primary el-button--small" id="zhxbutton" onclick="var row=document.getElementById('zhxanswer');var button=document.getElementById('zhxbutton');row.style.display='block';button.style.display='none';">
    <span>
        显示答案
    </span>
</button>
<div id="zhxanswer" style="display:none">
    <div class="row">
        答案：%s
    </div>
    <div class="row">
        %s
    </div>
</div>
</font>""" % (source, idx, question["statement"], question["option_list"]["A"], question["option_list"]["B"],
              question["option_list"]["C"], question["option_list"]["D"], answer, analyze)

    res["description"] = description
    res["content"] = "请仔细阅读题目，利用搜索网站搜索考点，并对每个选项所涉及的考点进行标注。"
    if not (label is None):
        res["answer"] = label

    return res


data = {
    "name": "司法考试考点标注",
    "description": "司法考试考点标注",
    "options": [],
    "extra": {
        "health": True,
        "healthType": 1,
        "limitNum": 0,
        "stopTime": 120,
        "first": True,
        "healthInterval": 30,
        "healthList": [
            generate_question("1__train", 1, {
                "statement": "我国民事诉讼法规定的管辖中，哪些管辖属于裁定管辖？",
                "option_list": {
                    "A": "级别管辖",
                    "B": "移送管辖",
                    "C": "指定管辖",
                    "D": "管辖权转移"
                },
                "analyse": "本题考查法定管辖与裁定管辖的区分。根据管辖是由法律直接规定还是由法院裁定确定为标准，可以将管辖分为法定管辖与裁定管辖。级别管辖和地域管辖均由民事诉讼法直接规定，故为法定管辖：移送管辖、指定管辖和管辖权移转虽然也在民事诉讼法都有规定，但它们的适用需要通过法院的裁定来实现，所以为裁定管辖。故BCD为正确选项。",
                "answer": ["B", "C", "D"]
            }, label=["9-3-5"]),
            generate_question("1__train", 2, {
                "statement": "在民事诉讼中，下列哪些人可以作为委托代理人？",
                "option_list": {
                    "A": "受过刑事处罚的人",
                    "B": "限制行为能力的人",
                    "C": "可能损害被代理人利益的人",
                    "D": "人民法院认为不宜作诉讼代理人的人"
                },
                "analyse": "本题的直接法律依据是《民诉意见》第68条，无民事行为能力人、限制民事行为能力人、可能损害被代理人利益的人以及法院认为不适宜作诉讼代理人的人均不能接受他人委托担任诉讼代理人。其实也可根据常理推断，对A项，可考虑不区分实际情况，将所有受过刑事处罚的人排除在外显然不合理，对D项，是立法者惯用的兜底条款，用以加强法律的完备性。",
                "answer": ["A"]
            }, label=["9-6-3-2"]),
            generate_question("1__train", 3, {
                "statement": "《大清民律草案》的亲属和继承两篇由哪几个机构共同起草?",
                "option_list": {
                    "A": "谘议局",
                    "B": "资政院",
                    "C": "修订法律馆",
                    "D": "礼学馆"
                },
                "analyse": "本题涉及晚清法律改革。亲属继承篇由修订法律馆和礼学馆共同起草，因为涉及“礼”。",
                "answer": ["C", "D"]
            }, label=["16-4-2-2-2"]),
        ]
    },
    "instructions": "http://103.242.175.80:16018/note",
    "type": {
        "type": 2,
        "multiple": True,
        "multipleLimit": 2
    },
    "level": 1,
    "repeat": 1,
    "questionList": [
    ]
}


def solve_file(filename, name):
    task = name.replace(".json", "")
    f = open(filename, "r")

    cnt = 0
    added = 0
    for line in f:
        x = json.loads(line)
        cnt += 1

        if not ("answer" in x.keys()):
            continue
        if not ("analyse" in x.keys()):
            continue

        y = generate_question(task, cnt, x)
        added += 1

        data["questionList"].append(y)

        if added == 15:
            break

    f.close()


if __name__ == "__main__":
    for filename in os.listdir(data_path):
        solve_file(os.path.join(data_path, filename), filename)

    json.dump(data, open("biao2.json", "w"), indent=2, ensure_ascii=False, sort_keys=True)
