import json

data = json.load(open("data.json", "r", encoding="utf8"))


def dfs_search(data, pre):
    s = ""

    cnt = 0
    for x in data:
        cnt += 1
        now_id = pre + "-" + str(cnt)

        if len(x["content"]) == 0:
            s = s + """ <a href="/get?id=%s" target="_blank">%s(最底层)</a><br>""" % (x["_id"], x["_name"] + " " + x["_id"])
        else:
            sp = """
<div class="card"><div class="card-header" id="%s"><h5 class="mb-0">
<button class="btn btn-link collapsed" type="button" data-toggle="collapse" data-target="#%s" aria-expanded="false" aria-controls="%s">
%s
 </button></h5></div>
<div id="%s" class="collapse" aria-labelledby="%s" data-parent="#%s">
<div class="card-body">""" % (
                now_id + "-head", now_id + "-collapse", now_id + "-collapse", x["_name"] + " " + x["_id"],
                now_id + "-collapse", now_id + "-head", pre)
            s = s + sp + dfs_search(x["content"], now_id + "-collapse")
            s = s + """
</div></div></div>
"""

    return s


if __name__ == "__main__":
    cnt = 0
    s = dfs_search(data, "home")

    print("""
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>xcjの查看网站</title>
</head>
<link href="/static/css/bootstrap.min.css" rel="stylesheet">
<script src="/static/js/jquery.min.js"></script>
<script src="/static/js/bootstrap.min.js"></script>
<script>
$(document).ready(function() {window.location.hash = "#home";})
</script>
<body>
<div class="container">
    <table id="FakeTable0" class="table table-striped">
    </table>
    <div class="row">
        <a href="/search">跳转至搜索页面</a>
    </div>
    <table id="FakeTable1" class="table table-striped">
    </table>

    <div class="accordion" id="home">
        %s
    </div>
</body>
</html>
""" % s, file=open("templates/view.html", "w", encoding="utf8"))
