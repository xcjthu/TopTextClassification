import argparse
import os
import torch
from torch import nn
import shutil
import json

from config_reader.parser import ConfigParser
from model.get_model import get_model
from reader.reader import init_formatter
from model.loss import get_loss

task_list = ["divorce", "labor", "loan"]

if __name__ == "__main__":
    for task in task_list:
        if os.path.exists(os.path.join("trained", task)):
            configFilePath = os.path.join("trained", task, "config")
            config = ConfigParser(configFilePath)

            model_name = config.get("model", "name")
            net = get_model(model_name, config)

            net.cuda()

            net.load_state_dict(torch.load(os.path.join("trained", task, "model.pkl")))

            net.eval()

            criterion = get_loss(config.get("train", "type_of_loss"))

            label = []
            with open(config.get("data", "label_file"), "r") as f:
                for line in f:
                    label.append(line[:-1])

            init_formatter(config)
            from reader.reader import formatter

            result = []
            with open(os.path.join("/input", task, "input.json"), "r") as f:
                for line in f:
                    data = json.loads(line)

                    arr = []

                    for x in data:
                        temp = {"sentence": str(x["sentence"]), "label": []}
                        data = formatter.format([temp])
                        result = net.forward(data, criterion, config, True)["result"]

                        for a in range(0, len(result[0])):
                            if result[0][a] > 0.5:
                                temp["label"].append(label[a])

                        arr.append(temp)
                    result.append(arr)

                with open(os.path.join("/output", task, "output.json"), "r") as f:
                    for x in result:
                        print(json.dumps(x, ensure_ascii=False, sort_keys=True), file=f)

    os.system("rm * -rf")
