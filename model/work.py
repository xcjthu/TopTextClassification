import os
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import json
from torch.optim import lr_scheduler
# from tensorboardX import SummaryWriter
import shutil
from timeit import default_timer as timer
from pytorch_pretrained_bert1 import BertAdam

from model.loss import get_loss
from utils.util import gen_result, print_info, time_to_str


def resulting(net, valid_dataset, use_gpu, config):
    net.eval()

    task_loss_type = config.get("train", "type_of_loss")
    criterion = get_loss(task_loss_type)

    running_acc = 0
    running_loss = 0
    cnt = 0
    acc_result = []

    result = []

    while True:
        data = valid_dataset.fetch_data(config)
        # print('fetch data')
        if data is None:
            break
        cnt += 1
        # print(data["label"])
        # gg

        with torch.no_grad():
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if torch.cuda.is_available() and use_gpu:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

        results = net(data, criterion, config, use_gpu, acc_result)

        for a in range(0, len(results["result"])):
            result.append(int(results["result"][a]))

        # print('forward')

        outputs, loss, accu = results["x"], results["loss"], results["accuracy"]
        acc_result = results["accuracy_result"]

        running_loss += loss.item()
        running_acc += accu.item()

    # print_info("Valid result:")
    # print_info("Average loss = %.5f" % (running_loss / cnt))
    # print_info("Average accu = %.5f" % (running_acc / cnt))
    # gen_result(acc_result, True)

    net.train()

    return result


def valid_net(net, valid_dataset, use_gpu, config, epoch, writer=None):
    net.eval()

    task_loss_type = config.get("train", "type_of_loss")
    criterion = get_loss(task_loss_type)

    running_loss = 0
    running_acc = {'law': 0, 'charge': 0, 'time': 0}
    cnt = 0
    acc_result = {'law': [], 'charge': [], 'time': []}

    #doc_list = []
    while True:
        data = valid_dataset.fetch_data(config)
        # print('fetch data')
        if data is None:
            break
        cnt += 1

        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if torch.cuda.is_available() and use_gpu:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        results = net(data, criterion, config, use_gpu, acc_result)
        # print('forward')

        outputs, loss, accu = results["x"], results["loss"], results["accuracy"]
        acc_result = results["accuracy_result"]

        #doc_list += results['doc_choice'].tolist()

        running_loss += loss.item()
        running_acc['law'] += accu['law'].item()
        running_acc['charge'] += accu['charge'].item()
        running_acc['time'] += accu['time'].item()

    '''
    print('law')
    gen_result(acc_result['law'], True)
    print('\n\ncharge')
    gen_result(acc_result['charge'], True)
    print('\n\ntime')
    gen_result(acc_result['time'], True)
    '''


    if writer is None:
        pass
    else:
        writer.add_scalar(config.get("output", "model_name") + " valid loss", running_loss / cnt, epoch)
        writer.add_scalar(config.get("output", "model_name") + " valid law accuracy", running_acc['law'] / cnt, epoch)
        writer.add_scalar(config.get("output", "model_name") + " valid charge accuracy", running_acc['charge'] / cnt, epoch)
        writer.add_scalar(config.get("output", "model_name") + " valid time accuracy", running_acc['time'] / cnt, epoch)

    # print_info("Valid result:")
    # print_info("Average loss = %.5f" % (running_loss / cnt))
    # print_info("Average accu = %.5f" % (running_acc / cnt))
    # gen_result(acc_result, True)

    net.train()

    for key in running_acc.keys():
        running_acc[key] /= cnt

    return running_loss / cnt, running_acc

    # print_info("valid end")
    # print_info("------------------------")


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

    # writer = SummaryWriter(
    #    os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
    #    config.get("output", "model_name"))
    writer = None

    criterion = get_loss(task_loss_type)

    optimizer_type = config.get("train", "optimizer")
    if optimizer_type == "adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                               weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=config.getfloat("train", "momentum"),
                              weight_decay=config.getfloat("train", "weight_decay"))
    elif optimizer_type == "bert_adam":
        optimizer = BertAdam(net.parameters(), lr=learning_rate, weight_decay=config.getfloat("train", "weight_decay"))
    else:
        raise NotImplementedError

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "gamma")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print('** start training here! **')
    print('----------------|----------------TRAIN----------------------|-----------------VALID---------------------|----------------|')
    print('  lr    epoch   |   loss       law       charge      time   |   loss       law       charge      time   |    run_time    |')
    print('----------------|-------------------------------------------|-------------------------------------------|----------------|')
    start = timer()

    for epoch_num in range(trained_epoch, epoch):
        cnt = 0
        total = 0

        train_cnt = 0
        train_loss = 0
        train_acc = {'law': 0, 'charge': 0, 'time': 0}

        exp_lr_scheduler.step(epoch_num)
        lr = 0
        for g in optimizer.param_groups:
            lr = float(g['lr'])
            break

        while True:
            cnt += 1
            data = train_dataset.fetch_data(config)
            if data is None:
                break

            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if torch.cuda.is_available() and use_gpu:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            optimizer.zero_grad()

            results = net(data, criterion, config, use_gpu)

            outputs, loss, accu = results["x"], results["loss"], results["accuracy"]

            loss.backward()
            train_loss += loss.item()
            train_acc['law'] += accu['law'].item()
            train_acc['charge'] += accu['charge'].item()
            train_acc['time'] += accu['time'].item()
            train_cnt += 1

            loss = loss.item()
            for key in accu:
                accu[key] = accu[key].item()
            #accu = accu.item()
            optimizer.step()

            total += config.getint("train", "batch_size")

            if cnt % output_time == 0:
                print('\r', end='', flush=True)
                print('%.4f   % 3d    |  %.4f    % 2.2f     % 2.2f     % 2.2f   |   ?????      ?????      ?????     ?????   |  %s  |' % (
                    lr, epoch_num + 1, train_loss / train_cnt, train_acc['law'] / train_cnt * 100, train_acc['charge'] / train_cnt * 100, train_acc['time'] / train_cnt * 100,
                    time_to_str((timer() - start))), end='',
                      flush=True)
            if cnt >= 5000:
                break

        del data

        train_loss /= train_cnt
        train_acc['law'] /= train_cnt
        train_acc['charge'] /= train_cnt
        train_acc['time'] /= train_cnt

        # writer.add_scalar(config.get("output", "model_name") + " train loss", train_loss, epoch_num + 1)
        # writer.add_scalar(config.get("output", "model_name") + " train accuracy", train_acc, epoch_num + 1)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(net.state_dict(), os.path.join(model_path, "model-%d.pkl" % (epoch_num + 1)))

        with torch.no_grad():
            valid_loss, valid_accu = valid_net(net, valid_dataset, use_gpu, config, epoch_num + 1, writer)
        print('\r', end='', flush=True)
        print('%.4f   % 3d    |  %.4f     %.2f      %.2f      %.2f   |  %.4f     % 2.2f     % 2.2f    % 2.2f   |  %s  |' % (
            lr, epoch_num + 1, train_loss, train_acc['law'] * 100, train_acc['charge'] * 100, train_acc['time'] * 100, 
            valid_loss, valid_accu['law'] * 100, valid_accu['charge'] * 100, valid_accu['time'] * 100,
            time_to_str((timer() - start))))


print_info("training is finished!")
