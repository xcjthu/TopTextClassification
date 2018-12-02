import json
import os

content = "担任法官"
path = r"data"


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
