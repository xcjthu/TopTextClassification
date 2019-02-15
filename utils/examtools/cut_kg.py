import jieba
import os
import json
import requests
from requests.auth import HTTPBasicAuth

input_data_path = "data2"
output_data_path = "cut_data2"

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
    data = json.load(open(os.path.join(input_data_path, path), "r"))

    for a in range(0, len(data)):
        data[a] = cut(data[a])

    json.dump(data, open(os.path.join(output_data_path, path), "w"))


def dfs_search(path):
    real_path = os.path.join(input_data_path, path)
    for filename in os.listdir(real_path):
        file_path = os.path.join(real_path, filename)
        if os.path.isdir(file_path):
            dfs_search(os.path.join(path, filename))
        else:
            cut_file(os.path.join(path, filename))


dfs_search("")
