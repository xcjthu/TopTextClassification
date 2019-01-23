import os
import json
import win32com
from docx import Document

doc_path = r"docx"
output_dir = "data2"

def parse_file(content):
    global cnt
    cnt += 1
    # print(cnt)
    result = []
    text = []
    for x in content:
        if len(x.text) > 0:
            text.append(x.text)
    print(text)
    gg
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
