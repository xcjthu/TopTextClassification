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


def calc_accuracy(outputs, label, config, result=None):
    if config.get("train", "type_of_loss") in multi_label:
        if len(label[0]) != len(outputs[0]):
            raise ValueError('Input dimensions of labels and outputs must match.')

        outputs = outputs.data
        labels = label.data
        
        if result is None:
            result = []

        total = 0
        nr_classes = outputs.size(1)

        while (len(result) < nr_classes):
            result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

        for i in range(nr_classes):
            outputs1 = (outputs[:, i] >= 0.5).long()
            labels1 = (labels[:, i] >= 0.5).long()
            total += int((labels1 * outputs1).sum())
            total += int(((1 - labels1) * (1 - outputs1)).sum())

            if result is None:
                continue

            # if len(result) < i:
            #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

            result[i]["TP"] += int((labels1 * outputs1).sum())
            result[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
            result[i]["FP"] += int(((1 - labels1) * outputs1).sum())
            result[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())

        return torch.Tensor([1.0 * total / len(outputs) / len(outputs[0])]), result
    else:

        if not (result is None):
            #print(label)
            id1 = torch.max(outputs, dim=1)[1]
            #id2 = torch.max(label, dim=1)[1]
            id2 = label
            nr_classes = outputs.size(1)
            while len(result) < nr_classes:
                result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
            for a in range(0, len(id1)):
                # if len(result) < a:
                #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

                it_is = int(id1[a])
                should_be = int(id2[a])
                if it_is == should_be:
                    result[it_is]["TP"] += 1
                else:
                    result[it_is]["FP"] += 1
                    result[should_be]["FN"] += 1
        pre, prediction = torch.max(outputs, 1)
        prediction = prediction.view(-1)

        return torch.mean(torch.eq(prediction, label).float()), result


def get_value(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def gen_result(res, print=False):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_value(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_value(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    if print:
        print_info("Micro precision\t%.3f" % micro_precision)
        print_info("Macro precision\t%.3f" % macro_precision)
        print_info("Micro recall\t%.3f" % micro_recall)
        print_info("Macro recall\t%.3f" % macro_recall)
        print_info("Micro f1\t%.3f" % micro_f1)
        print_info("Macro f1\t%.3f" % macro_f1)
