import jieba
import os
import json
import requests
from requests.auth import HTTPBasicAuth

input_data_path = "/data/disk3/private/zhx/exam/data/origin_data/no_subject"
output_data_path = "/data/disk3/private/zhx/exam/data/cut_data/no_subject"

word_set = set()

username = None
password = None


def cut(content):
    global username, password
    if password is None:
        print("Enter username: ", end='')
        username = input().replace("\n", "")
        print("Enter password: ", end='')
        password = input().replace("\n", "")

    url = "http://114.112.106.221:9200/_analyze"
    response = requests.get(url, data=json.dumps({"analyzer": "ik_smart", "text": content}),
                            auth=HTTPBasicAuth(username, password),
                            headers={"Content-Type": "application/json"})

    data = json.loads(response.text)["tokens"]
    content = []
    for x in data:
        word = x["token"]
        content.append(word)
        word_set.add(word)

    return content


def cut_file(path):
    print(path)
    os.makedirs(os.path.join(output_data_path, "/".join(path.split("/")[:-1])), exist_ok=True)
    input_file = open(os.path.join(input_data_path, path), "r")
    output_file = open(os.path.join(output_data_path, path), "w")

    for line in input_file:
        try:
            data = json.loads(line)
            data["statement"] = cut(data["statement"].replace("\n", ""))

            for option in data["option_list"]:
                data["option_list"][option] = cut(data["option_list"][option].replace("\n", ""))

            data["subject"] = 0

            print(json.dumps(data, ensure_ascii=False, sort_keys=True), file=output_file)

        except Exception as e:
            print(e)
            raise e

    input_file.close()
    output_file.close()


def dfs_search(path):
    real_path = os.path.join(input_data_path, path)
    for filename in os.listdir(real_path):
        file_path = os.path.join(real_path, filename)
        if os.path.isdir(file_path):
            dfs_search(os.path.join(path, filename))
        else:
            cut_file(os.path.join(path, filename))


dfs_search("")
