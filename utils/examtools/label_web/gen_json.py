import json
import os

data_path = "/data/disk3/private/zhx/exam/data/origin_data/final4"

data = {
    "name": "司法考试考点标注",
    "description": "司法考试考点标注",
    "options": [],
    "extra": {
        "health": True,
        "healthType": 0,
        "limitNum": 0,
        "stopTime": 120,
        "first": True,
        "healthInterval": 2,
        "healthList": [
            {
                "description": "题目来源：1__train 题目编号：1 题目选项：A\n" + " <a href=\"http://103.242.175.80:16018/search\" target=\"_blank\" style=\"color:#55A2F3\">跳转至搜索网站</a>\n" + "问题描述",
                "content": "题目：元代人在《唐律疏议序》中说：“乘之（指唐律）则过，除之则不及，过与不及，其失均矣。”表达了对唐律的敬畏之心。下列关于唐律的哪一表述是错误的？\n" + "选项A：促使法律统治“一准乎礼”，实现了礼律统一\n" + "解析：选项A表述正确。唐朝承袭和发展了以往礼法并用的统治方法，使得法律统治“一准乎礼”，真正实现了礼与律的统一。",
                "answer": ["16-3-1-2-1"]
            }
        ]
    },
    "instructions": "https://powerlawai.oss-cn-beijing.aliyuncs.com/static/%E8%A7%85%E5%BE%8Blogo_108_108.png",
    "type": {
        "type": 2,
        "multiple": True,
        "multipleLimit": 2
    },
    "level": 0,
    "repeat": 3,
    "questionList": [
    ]
}


def solve_file(filename, name):
    task = name.replace(".json", "")
    f = open(filename, "r")

    cnt = 0
    for line in f:
        x = json.loads(line)
        z = []
        cnt += 1

        for option in ["A", "B", "C", "D"]:
            y = {}
            y["description"] = "题目来源：%s 题目编号：%d 题目选项：%s\n" % (task, cnt,
                                                              option) + " <a href=\"http://103.242.175.80:16018/search\" target=\"_blank\" style=\"color:#55A2F3\">跳转至搜索网站</a>\n" + "问题描述"
            y["content"] = "题目：" + x["statement"] + ("\n选项%s：" % option) + x["option_list"][option]
            if "analyze" in x.keys():
                y["content"] += "\n解析：" + y["analyze"]
            z.append(y)
        data["questionList"].append(z)

        if cnt == 1500:
            break

    f.close()


if __name__ == "__main__":
    for filename in os.listdir(data_path):
        solve_file(os.path.join(data_path, filename), filename)

    json.dump(data, open("biao.json", "w"), indent=2, ensure_ascii=False, sort_keys=True)
