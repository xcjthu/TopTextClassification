import os
import json
import win32com
from docx import Document

doc_path = r"docx"
output_dir = "data"

result = []

number_dic = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "零": 0
}


def is_level5(text):
    if not (text[0] in ["(", "（"]):
        return False
    p = 1
    while text[p] in number_dic.keys():
        p += 1
    if p == 1:
        return False
    if text[p] in [")", "）"]:
        return True
    return False


def is_level4(text):
    p = 0
    while text[p] in number_dic.keys():
        p += 1
    if p == 0:
        return False
    if text[p] in ["、"]:
        return True
    return False


def is_level3(text):
    if text[0] != "第":
        return False
    p = 0
    for p in range(1, len(text)):
        if text[p] == "节":
            for a in range(1, p):
                if not (text[a] in number_dic.keys()):
                    return False
            return True

    return False


def is_level2(text):
    if text[0] != "第":
        return False
    p = 0
    for p in range(1, len(text)):
        if text[p] == "章":
            for a in range(1, p):
                if not (text[a] in number_dic.keys()):
                    return False
            return True

    return False


def parse(text, p=0, now_level=2):
    res = []
    while p < len(text):
        tmp = text[p]
        if is_level3(tmp):
            if now_level >= 3:
                return res, p - 1
            else:
                x, y = parse(text, p + 1, 3)
                res.append({
                    "_name": tmp,
                    "_level": 3,
                    "content": x
                })
                p = y
        elif is_level4(tmp):
            if now_level >= 4:
                return res, p - 1
            else:
                x, y = parse(text, p + 1, 4)
                res.append({
                    "_name": tmp,
                    "_level": 4,
                    "content": x
                })
                p = y
        elif is_level5(tmp):
            if now_level >= 5:
                return res, p - 1
            else:
                x, y = parse(text, p + 1, 5)
                res.append({
                    "_name": tmp,
                    "_level": 5,
                    "content": x
                })
                p = y
        else:
            res.append(tmp)
        p += 1
    return res, p


def parse_file(filepath):
    document = Document(filepath)
    content = document.paragraphs

    text = []
    for x in content:
        if len(x.text) > 0:
            s = x.text
            s = s.replace("\t", " ").split("\n")
            for y in s:
                while len(y) > 0 and y[0] == " ":
                    y = y[1:]
                if len(y) == 0:
                    continue
                text.append(y)

    for a in range(0, len(text)):
        if is_level2(text[a]):
            if text[a][-1] == "章":
                return {
                    "_name": text[a] + " " + text[a + 1],
                    "_level": 2,
                    "content": parse(text, a + 2)[0]
                }
            else:
                return {
                    "_name": text[a],
                    "_level": 2,
                    "content": parse(text, a + 1)[0]
                }

    gg
    return parse(text)


def get_number_from_s(s):
    v = 0
    first = True
    for a in range(0, len(s)):
        if s[a] in number_dic.keys():
            if s[a] == "十":
                if first:
                    v = 1
                v = v * 10
            else:
                v = v + number_dic[s[a]]
            first = False
        elif s[a] == "章":
            break
    return v


def sort_array(arr):
    for a in range(0, len(arr)):
        for b in range(a + 1, len(arr)):
            if get_number_from_s(arr[a]) > get_number_from_s(arr[b]):
                arr[a], arr[b] = arr[b], arr[a]

    return arr


if __name__ == "__main__":
    name1 = sort_array(os.listdir(doc_path))
    # name1.sort()
    for dir_name in name1:
        name2 = os.listdir(os.path.join(doc_path, dir_name))
        name2.sort()
        for level1_name in name2:
            temp = {
                "_name": level1_name,
                "_level": 1,
                "content": []
            }

            name3 = sort_array(os.listdir(os.path.join(doc_path, dir_name, level1_name)))
            for x in name3:
                print(x, get_number_from_s(x))
            # name3.sort()
            for level2_name in name3:
                temp["content"].append(parse_file(os.path.join(doc_path, dir_name, level1_name, level2_name)))

            result.append(temp)

    # print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    json.dump(result, open("content.json", "w", encoding="utf8"), indent=2, ensure_ascii=False, sort_keys=True)
