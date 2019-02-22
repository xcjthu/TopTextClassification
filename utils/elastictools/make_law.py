import json
import os
import time
import multiprocessing
import uuid

law_list = set(json.load(open("/home/zhx/law_name.json", "r")))


def print_time():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def print_info(s):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("[%s] %s" % (times, s), flush=True)


result = []


def insert_doc(x):
    result.append(x)


num_list = {
    u"〇": 0,
    u"\uff2f": 0,
    u"\u3007": 0,
    u"\u25cb": 0,
    u"\uff10": 0,
    u"\u039f": 0,
    u'零': 0,
    "O": 0,
    "0": 0,
    u"一": 1,
    u"元": 1,
    u"1": 1,
    u"二": 2,
    u"2": 2,
    u'三': 3,
    u'3': 3,
    u'四': 4,
    u'4': 4,
    u'五': 5,
    u'5': 5,
    u'六': 6,
    u'6': 6,
    u'七': 7,
    u'7': 7,
    u'八': 8,
    u'8': 8,
    u'九': 9,
    u'9': 9,
    u'十': 10,
    u'百': 100,
    u'千': 1000,
    u'万': 10000
}


def get_number_from_string(s):
    for x in s:
        if not (x in num_list):
            print(s)
            gg

    value = 0
    try:
        value = int(s)
    except ValueError:
        nowbase = 1
        addnew = True
        for a in range(len(s) - 1, -1, -1):
            if s[a] == u'十':
                nowbase = 10
                addnew = False
            elif s[a] == u'百':
                nowbase = 100
                addnew = False
            elif s[a] == u'千':
                nowbase = 1000
                addnew = False
            elif s[a] == u'万':
                nowbase = 10000
                addnew = False
            else:
                value = value + nowbase * num_list[s[a]]
                addnew = True

        if not (addnew):
            value += nowbase

    return value


def parse(s):
    s = s.replace("\t", "").replace("\n", "").replace("|", "").replace("\u3000", "").replace(r"\u3000", "")
    res = []
    temps = ""

    pre = 0

    a = 0
    while a < len(s):
        if s[a] == "第":
            b = a + 1
            while b < len(s) and s[b] in num_list.keys():
                b += 1
            if s[b] == "条":
                x = get_number_from_string(s[a + 1:b])
                if x == pre + 1:
                    pre += 1
                    if len(temps) != 0 and len(temps) < 500:
                        res.append(temps)
                    temps = s[a:b + 1]
                    a = b
                else:
                    temps = temps + s[a]
            else:
                temps = temps + s[a]

        else:
            temps = temps + s[a]
        a += 1

    if len(temps) != 0 and len(temps) < 500:
        res.append(temps)

    return res


def dfs(x):
    z = x["article_content"]
    for y in x["child"]:
        z = z + " " + dfs(y)
    return z


def dfs_insert(index, doc_type, title, x):
    if x["article_level"] == 4:
        z = dfs(x)
        insert_doc(z)
    else:
        for y in x["child"]:
            dfs_insert(index, doc_type, title, y)


def insert_file(index, doc_type, file_path):
    data = json.load(open(file_path, "r"))

    total = len(data["laws"])
    cnt = 0

    for x in data["laws"]:
        if not (x["authority_level"] in [1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18]):
            pass
        elif x["title"].find("批复") != -1:
            pass
        elif x["title"].find("答复") != -1:
            pass
        elif x["title"].find("复函") != -1:
            pass
        elif not (x["title"] in law_list):
            pass
        else:
            if len(x["law_articles"]) == 0:
                z = parse(x["content"])

                for y in z:
                    insert_doc(y)
            else:
                for y in x["law_articles"]:
                    dfs_insert(index, doc_type, x["title"], y)
        cnt += 1

        print('\r', end='', flush=True)
        print("%d/%d" % (cnt, total), end='', flush=True)

    print("")


if __name__ == "__main__":
    file_list = []

    for a in range(0, 9):
        file_list.append("a%d_new.json" % a)
    for a in range(0, 5):
        file_list.append("b%d_new.json" % a)
    pre_path = "/data/disk1/data/law_data/"

    for name in file_list:
        name = pre_path + name
        print(name)
        insert_file("?", "?", name)

    json.dump(result, "/home/zhx/law_content.json", indent=2, ensure_ascii=False, sort_keys=True)
