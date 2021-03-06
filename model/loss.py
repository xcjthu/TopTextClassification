import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_loss(task_loss_type):
    if task_loss_type == "cross_entropy_loss":
        criterion = cross_entropy_loss
    elif task_loss_type == "nll_loss":
        criterion = nn.NLLLoss()
    elif task_loss_type == "focal_loss":
        criterion = FocalLoss()
    elif task_loss_type == "multi_label_cross_entropy_loss":
        criterion = multi_label_cross_entropy_loss
    else:
        raise NotImplementedError

    return criterion


def multi_label_cross_entropy_loss(outputs, labels):
    temp = F.sigmoid(outputs)
    res = - labels * torch.log(temp) - (1 - labels) * torch.log(1 - temp)
    res = torch.mean(torch.sum(res, dim=1))

    return res


def cross_entropy_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
