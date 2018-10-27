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


def calc_accuracy(outputs, label):
    pre, prediction = torch.max(outputs, 1)
    prediction = prediction.view(-1)
    return torch.mean(torch.eq(prediction, label).float())
