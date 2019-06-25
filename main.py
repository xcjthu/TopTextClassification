import argparse
import os
import torch
from torch import nn
from torch.autograd import Variable
import shutil
import json

from config_reader.parser import ConfigParser
from model.get_model import get_model
from reader.reader import init_formatter
from model.loss import get_loss

task_list = ["divorce", "labor", "loan"]

if True:
    for task in task_list:
        if os.path.exists(os.path.join("trained", task)):
            configFilePath = os.path.join("trained", task, "config")
            config = ConfigParser(configFilePath)

            model_name = config.get("model", "name")
            net = get_model(model_name, config)

            net.cuda()
            net.init_multi_gpu([0])

            net.load_state_dict(torch.load(os.path.join("trained", task, "model.pkl")))

            net.eval()

            criterion = get_loss(config.get("train", "type_of_loss"))

            label = []
            with open(config.get("data", "label_file"), "r") as f:
                for line in f:
                    label.append(line[:-1])

            init_formatter(config)
            from reader.reader import formatter

            res = []
            with open(os.path.join("/input", task, "input.json"), "r", encoding="utf8") as f:
                for line in f:
                    data = json.loads(line)

                    arr = []

                    for x in data:
                        temp = {"sentence": str(x["sentence"]), "labels": []}
                        data = formatter.format([temp], config, None, None)
                        for key in data.keys():
                            if isinstance(data[key], torch.Tensor):
                                data[key] = Variable(data[key].cuda())
                        result = net.forward(data, criterion, config, True)["result"]

                        for a in range(0, len(result[0])):
                            if result[0][a] > 0.5:
                                temp["labels"].append(label[a])

                        arr.append(temp)
                    res.append(arr)

                with open(os.path.join("/output", task, "output.json"), "w", encoding="utf8") as f:
                    for x in res:
                        print(json.dumps(x, ensure_ascii=False, sort_keys=True), file=f)

    os.system("rm * -rf")
