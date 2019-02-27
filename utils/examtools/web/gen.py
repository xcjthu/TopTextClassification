import json

data = json.load(open("/data/disk3/private/zhx/exam/data/solve/x.json", "r"))

if __name__ == "__main__":
    content = ""

    for a in range(0, len(data)):
        if data[a]["type"]%2==0:
            s = "单选"
        else:
            s = "多选"
        content = content + """
%d.(%s) %s
   A. %s 
   B. %s
   C. %s
   D. %s

        """ % (a + 1, s, data[a]["statement"],
               data[a]["option_list"]["A"],
               data[a]["option_list"]["B"],
               data[a]["option_list"]["C"],
               data[a]["option_list"]["D"])

    print(content, file=open("problem.md", "w"))
