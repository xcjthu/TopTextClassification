import os
import json
import win32com
from win32com import client as wc

doc_path = r"D:\university\work\nlp\law\司法考试大纲\司法考试大纲"
output_dir = "data"


def solve_file(filepath):
    print(filepath)
    outfilepath = os.path.join(output_dir, filepath)
    filepath = os.path.join(doc_path, filepath)

    os.makedirs("\\".join(outfilepath.split("\\")[:-1]), exist_ok=True)

    if filepath.split(".")[-1] == "doc":
        word = wc.Dispatch('Word.Application')
        doc = word.Documents.Open(filepath)
        doc.SaveAs(filepath + "x", 12, False, "", False, "", False, False, False, False)
    elif filepath.split(".")[-1] == "docx":
        pass
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
