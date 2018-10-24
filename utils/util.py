import time
import torch


def print_time():
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def print_info(s):
    times = str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("[%s] %s" % (times, s))


def get_file_list(file_name):
    return file_name.replace(" ", "").split(",")


def calc_accuracy(outputs, label):
    pre, prediction = torch.max(outputs, 1)
    prediction = prediction.view(-1)
    return torch.mean(torch.eq(prediction, label).float())
