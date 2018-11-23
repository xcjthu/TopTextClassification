import json
import os

data_path = "/data/disk3/private/zhx/exam/data/origin_data"

analyze_result = {
    "total": 0,
    "num": [0, 0, 0, 0, 0],
    "statement_len": {},
    "answer_len": {}
}

file_list = os.listdir(data_path)
for filename in file_list:
    print(filename)
    file = open(os.path.join(data_path, filename), "r")
    for line in file:
        d = json.loads(line)
        analyze_result["total"] += 1
        analyze_result["num"][len(d["answer"])] += 1

        l = len(d["statement"])
        if not (l in analyze_result["statement_len"].keys()):
            analyze_result["statement_len"][l] = 0
        analyze_result["statement_len"][l] += 1
        if not ("option_list" in d.keys()):
            d["option_list"] = d["option"]

        for option in d["option_list"]:
            l = len(d["option_list"][option])
            if not (l in analyze_result["answer_len"].keys()):
                analyze_result["answer_len"][l] = 0
            analyze_result["answer_len"][l] += 1

json.dump(analyze_result, open("analyze_result.txt", "w"), indent=2, sort_keys=True, ensure_ascii=False)
