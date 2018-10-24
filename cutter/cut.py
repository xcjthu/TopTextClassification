from thulac_cutter import Thulac
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--id', '-i')
args = parser.parse_args()
platform = int(args.id)

cutter = Thulac("/home/zhx/THULAC/models", "/home/zhx/THULAC/dict.txt")

num_process = 40
data_dir = "/data/disk1/data/wenshu_data/new_json_data"
output_dir = "/data/disk1/private/zhx/law/cutted_data"


def cut(file_path):
    inf = open(file_path, "r")
    output_path = os.path.join(output_dir, file_path.split("/")[-2])
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, file_path.split("/")[-1])

    ouf = open(output_path, "w")

    for line in inf:
        data = json.loads(line[:-1])

        data["WS"]["QW"]["@value"] = cutter.fast_cut(data["WS"]["QW"]["@value"])
        data["WS"]["SB"]["SSJL"]["@value"] = cutter.fast_cut(data["WS"]["SB"]["SSJL"]["@value"])
        data["WS"]["SS"]["@value"] = cutter.fast_cut(data["WS"]["SS"]["@value"])
        data["WS"]["LY"]["@value"] = cutter.fast_cut(data["WS"]["LY"]["@value"])
        data["WS"]["PJJG"]["@value"] = cutter.fast_cut(data["WS"]["PJJG"]["@value"])
        data["WS"]["WB"]["@value"] = cutter.fast_cut(data["WS"]["WB"]["@value"])
        data["WS"]["QTXX"]["TITLE"]["@value"] = cutter.fast_cut(data["WS"]["QTXX"]["TITLE"]["@value"])

        print(json.dumps(data, ensure_ascii=False), file=ouf)

    gg


def dfs_search(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            dfs_search(file_path)
        else:
            idx = int(file_name.split(".")[0])
            if idx % num_process == platform:
                cut(file_path)


if __name__ == "__main__":
    dfs_search(data_dir)
