import time
import torch
import os


def print_time():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def print_info(s):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("[%s] %s" % (times, s))


def get_file_list(file_path, file_name):
    file_list = file_name.replace(" ", "").split(",")
    for a in range(0, len(file_list)):
        file_list[a] = os.path.join(file_path, file_list[a])
    return file_list


multi_label = ["multi_label_cross_entropy_loss"]


def calc_accuracy(outputs, label, config):
    if config.get("train", "type_of_loss") in multi_label:
        if len(label[0]) != len(outputs[0]):
            raise ValueError('Input dimensions of labels and outputs must match.')

        outputs = outputs.data
        labels = label.data

        total = 0
        nr_classes = outputs.size(1)
        for i in range(nr_classes):
            outputs1 = (outputs[:, i] >= 0.5).long()
            labels1 = (labels[:, i] >= 0.5).long()
            total += int((labels1 * outputs1).sum())

        return torch.Tensor(1.0 * total / len(outputs) / len(outputs[0]))
    else:
        pre, prediction = torch.max(outputs, 1)
        prediction = prediction.view(-1)
        return torch.mean(torch.eq(prediction, label).float())
