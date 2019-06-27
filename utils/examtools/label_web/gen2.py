import json

data = json.load(open("../content.json", "r", encoding="utf8"))


def dfs_search(data, pre):
    cnt = 0
    for x in data:
        if type(x) is str:
            continue
        cnt += 1
        x["_id"] = pre + str(cnt)
        dfs_search(x["content"], pre + str(cnt) + "-")


if __name__ == "__main__":
    cnt = 0
    dfs_search(data, "")

    json.dump(data, open("data2.json", "w", encoding="utf8"), ensure_ascii=False, sort_keys=True, indent=2)