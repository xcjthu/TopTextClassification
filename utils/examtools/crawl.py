import json
import os
import urllib
import urllib3
import requests

http = urllib3.PoolManager()

output_path = "crawl_data/"
os.makedirs(output_path, exist_ok=True)

sessions = requests.session()
sessions.headers[
    'User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def crawl(word):
    url = "https://baike.baidu.com/item/%s" % (urllib.parse.urlencode({"": word})[1:])
    # print(url)

    res = http.request("GET", url)

    s = bytes.decode(res.data)

    print(s, file=open(os.path.join(output_path, "%s.html" % word), "w", encoding="utf8"))

    if s.find("百度百科错误页") != -1:
        return False

    return Trie


if __name__ == "__main__":
    word_list = json.load(open("/data/disk3/private/zhx/exam/data/dict/final_dict.txt", "r"))

    for word in word_list:
        word = word[0]
        if crawl(word):
            pass
        else:
            print(word)

        import random

        if random.randint(1, 20) == 1:
            gg
