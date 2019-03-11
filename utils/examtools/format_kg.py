import os
import json
import win32com
from docx import Document

doc_path = r"docx"
output_dir = "data"

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
}


def is_level1(x):
    if x[0] == "第" and x[2] == "节":
        return True

    return False


def is_level2(x):
    if (x[0] in number_dic.keys() or x[0] in "1234567890") and x[1] == "、":
        return True

    return False


def is_level3(x):
    if x[0] == "（" and x[2] == "）" and x[1] in number_dic.keys():
        return True

    return False


def is_level4(x):
    if x[0] in "1234567890" and (x[1] == "." or x[1] == "．"):
        return True

    return False


def parse_level4(text, a):
    result = []
    temp = [text[a]]
    a = a + 1
    while a < len(text):
        x = text[a]
        if is_level4(x) or is_level3(x) or is_level2(x) or is_level1(x):
            if len(temp) != 0:
                result.append(temp)
            return result, a

        if is_level4(x):
            pass
            x, a = parse_level4(text, a)
            a -= 1

        temp.append(x)
        a += 1

    if len(temp) != 0:
        result.append(temp)

    return result, a


def parse_level3(text, a):
    result = []
    temp = [text[a]]
    a = a + 1
    while a < len(text):
        x = text[a]
        if is_level3(x) or is_level2(x) or is_level1(x):
            if len(temp) != 0:
                result.append(temp)
            return result, a

        if is_level4(x):
            x, a = parse_level4(text, a)
            a -= 1

        temp.append(x)
        a += 1

    if len(temp) != 0:
        result.append(temp)

    return result, a


def parse_level2(text, a):
    result = []
    temp = [text[a]]
    a = a + 1
    while a < len(text):
        x = text[a]
        if is_level2(x) or is_level1(x):
            if len(temp) != 0:
                result.append(temp)
            return result, a

        if is_level3(x):
            x, a = parse_level3(text, a)
            a -= 1

        temp.append(x)
        a += 1

    if len(temp) != 0:
        result.append(temp)

    return result, a


def parse_level1(text):
    result = []
    temp = []
    a = 0

    while a < len(text):
        x = text[a]
        if is_level1(x):
            result.append(temp)
            temp = []

        if is_level2(x):
            x, a = parse_level2(text, a)
            a -= 1

        temp.append(x)
        a += 1

    if len(temp) != 0:
        result.append(temp)

    return result


cnt = 0


def parse_file(content):
    global cnt
    cnt += 1
    # print(cnt)
    result = []
    text = []
    for x in content:
        if len(x.text) > 0:
            text.append(x.text)
    result = parse_level1(text)
    # if cnt == 4:
    #    print(json.dumps(result, indent=2, ensure_ascii=False))
    #    gg
    return result


def solve_file(filepath):
    outfilepath = os.path.join(output_dir, filepath.replace("docx", "txt"))
    filepath = os.path.join(doc_path, filepath)

    os.makedirs("\\".join(outfilepath.split("\\")[:-1]), exist_ok=True)

    if filepath.split(".")[-1] == "doc":
        pass
    elif filepath.split(".")[-1] == "docx":
        # print(filepath)
        document = Document(filepath)
        result = parse_file(document.paragraphs)
        if len(result) == 0:
            print("\\".join(filepath.split("\\")[0:]))
        # print(filepath)
        json.dump(result, open(outfilepath, "w", encoding="utf8"), indent=2, ensure_ascii=False)
    else:
        raise NotImplementedError


def dfs_search(path):
    now_path = os.path.join(doc_path, path)
    for filename in os.listdir(now_path):
        filepath = os.path.join(now_path, filename)

        if os.path.isdir(filepath):
            dfs_search(os.path.join(path, filename))
        else:
            solve_file(os.path.join(path, filename))


if __name__ == "__main__":
    dfs_search("")
