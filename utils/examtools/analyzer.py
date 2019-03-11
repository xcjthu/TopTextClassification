import os
import json

input_data_path = "data2"

l = 0


def solve(filename):
    filename = os.path.join(input_data_path, filename)
    global l
    l += len(json.load(open(filename, "r", encoding="utf8")))


def dfs_search(path):
    real_path = os.path.join(input_data_path, path)
    for filename in os.listdir(real_path):
        file_path = os.path.join(real_path, filename)
        if os.path.isdir(file_path):
            dfs_search(os.path.join(path, filename))
        else:
            solve(os.path.join(path, filename))


if __name__ == "__main__":
    dfs_search("")
    print(l)
