import os
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import json
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import shutil

from model.loss import get_loss
from utils.util import print_info


def valid_net(net, valid_dataset, use_gpu, config, epoch, writer=None):
    print_info("------------------------")
    print_info("valid begin")
    net.eval()

    task_loss_type = config.get("data", "type_of_loss")
    criterion = get_loss(task_loss_type)

    running_acc = 0
    running_loss = 0
    cnt = 0

    # TODO
    # Here to read data

    if writer is None:
        pass
    else:
        writer.add_scalar(config.get("output", "model_name") + " valid loss", running_loss / cnt, epoch)
        writer.add_scalar(config.get("output", "model_name") + " valid accuracy", running_acc / cnt, epoch)

    print_info("Valid result:")
    print_info("Average loss = %.5f" % (running_loss / cnt))
    print_info("Average accu = %.5f" % (running_acc / cnt))

    net.train()

    print_info("valid end")
    print_info("------------------------")


def train_net(net, train_dataset, valid_dataset, use_gpu, config):
    epoch = config.getint("train", "epoch")
    learning_rate = config.getfloat("train", "learning_rate")
    task_loss_type = config.get("train", "type_of_loss")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")
    model_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))

    try:
        trained_epoch = config.get("train", "pre_train")
        trained_epoch = int(trained_epoch)
    except Exception as e:
        trained_epoch = 0

    os.makedirs(os.path.join(config.get("output", "tensorboard_path")), exist_ok=True)

    if trained_epoch == 0:
        shutil.rmtree(
            os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")), True)

    writer = SummaryWriter(
        os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
        config.get("output", "model_name"))

    criterion = get_loss(task_loss_type)

    optimizer_type = config.get("train", "optimizer")
    if optimizer_type == "adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                               weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=config.getfloat("train", "momentum"),
                              weight_decay=config.getfloat("train", "weight_decay"))
    else:
        raise NotImplementedError

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "gamma")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print_info("training process start!")
    for epoch_num in range(trained_epoch, epoch):
        running_loss = 0
        running_acc = 0
        cnt = 0

        gb_cnt = 0
        gb_loss = 0
        gb_acc = 0

        exp_lr_scheduler.step(epoch_num)

        for g in optimizer.param_groups:
            print_info("Epoch %d, with learing rate %f" % (epoch_num + 1, float(g['lr'])))
            break

        # TODO
        # Here to read data to train

        writer.add_scalar(config.get("output", "model_name") + " train loss", gb_loss / gb_cnt, epoch_num + 1)
        writer.add_scalar(config.get("output", "model_name") + " train accuracy", gb_acc / gb_cnt, epoch_num + 1)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(net.state_dict(), os.path.join(model_path, "model-%d.pkl" % (epoch_num + 1)))

        if (epoch_num + 1) % test_time == 0:
            valid_net(net, valid_dataset, use_gpu, config, epoch_num + 1, writer)

    print_info("training is finished!")
