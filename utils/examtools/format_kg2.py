import os
import json
import win32com
from docx import Document

doc_path = r"docx"
output_dir = "data2"

cnt = 0

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


# (一)
# 一.
# 1.
# 第几节

def check_format(s):
    if len(s) <= 2:
        return None, None
    if s[-1] == ":":
        p = len(s) - 1
        while p >= 0:
            if s[p] == "。":
                break
            p -= 1
        p += 1
        return 4, s[p:]

    if s.find("。") != -1:
        return None, None

    if s[0] == "第" and s[1] in number_dic.keys() and s[2] == "节":
        return 3, s[3:]

    if s[0] == "(" and s[2] == ")" and s[1] in number_dic.keys():
        return 0, s[3:]

    if s[0] in number_dic.keys() and s[1] == ".":
        return 1, s[2:]

    if s[0] in "1234567890" and s[1] == ".":
        return 2, s[2:]

    return None, None


def parse(text):
    prefix = []
    result = []
    for s in text:
        s = s.replace("\n", "").replace(" ", "").replace("、", ".").replace("（", "(").replace("）", ")").replace("\t",
                                                                                                               "").replace(
            "：", ":")
        t, x = check_format(s)
        if not (t is None):
            for a in range(0, len(prefix)):
                if prefix[a][0] == t:
                    prefix = prefix[:a]
                    break

        if x is None or len(x) > 15:
            p = ""
            for a, b in prefix:
                p = p + b + " "
            p = p + s
            result.append(p)

        if not (t is None):
            prefix.append([t, x])

    return result


def parse_file(content):
    global cnt
    cnt += 1
    # print(cnt)
    result = []
    text = []
    for x in content:
        if len(x.text) > 0:
            text.append(x.text)

    text = parse(text)
    # print(json.dumps(text, ensure_ascii=False, sort_keys=True, indent=2))
    # if cnt == 50:
    #    print(json.dumps(text, indent=2, ensure_ascii=False))
    #    gg
    return result


def solve_file(filepath):
    print(filepath)
    outfilepath = os.path.join(output_dir, filepath.replace("docx", "txt"))
    filepath = os.path.join(doc_path, filepath)

    os.makedirs("\\".join(outfilepath.split("\\")[:-1]), exist_ok=True)

    if filepath.split(".")[-1] == "doc":
        pass
    elif filepath.split(".")[-1] == "docx":
        # print(filepath)
        document = Document(filepath)
        result = parse_file(document.paragraphs)
        # if len(result) == 0:
        #    print("\\".join(filepath.split("\\")[0:]))
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
