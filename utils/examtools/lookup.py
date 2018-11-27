import json
import os

content = [
    "全面依法治国，必须坚持人民的主体地位。对此，下列哪一理解是错误的？",
    "法律既是保障人民自身权利的有力武器，也是人民必须遵守的行为规范",
    "人民依法享有广泛的权利和自由，同时也承担应尽的义务",
    "人民通过各种途径直接行使立法、执法和司法的权力",
    "人民根本权益是法治建设的出发点和落脚点，法律要为人民所掌握、所遵守、所运用"
]
path = r"data\2018年国家法律职业考试辅导用书（一）"


def work(data, content):
    for x in data:
        if type(x) is list:
            if work(x, content):
                return True
        else:
            if x.find(content) != -1:
                return True

    return False


def lookup(filepath, content):
    data = json.load(open(filepath, "r", encoding="utf8"))
    if work(data, content):
        print(filepath)


def dfs_search(filepath):
    for filename in os.listdir(filepath):
        path = os.path.join(filepath, filename)
        if os.path.isdir(path):
            dfs_search(path)
        else:
            lookup(path, content)


if __name__ == "__main__":
    dfs_search(path)
