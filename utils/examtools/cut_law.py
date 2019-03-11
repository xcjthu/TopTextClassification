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


def cut_file():
    data = json.load(open("/home/zhx/law_content.json", "r"))

    for a in range(0, len(data)):
        data[a] = cut(data[a])

    json.dump(data, open("/home/zhx/law_cut_content.json", "w"), sort_keys=True, ensure_ascii=False, indent=2)


cut_file()
