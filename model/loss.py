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
    elif task_loss_type == "DSQA_loss":
        criterion = DSQA_loss
    else:
        raise NotImplementedError

    return criterion


'''
def EM_and_cross_entropy_loss(option_prob, option_output, labels):
    loss_cross = cross_entropy_loss(option_output, labels)
    label_one_shot = torch.zeros(option_prob.size()).cuda()
    label_one_shot.scatter_(dim = 1, index = labels.unsqueeze(1), value = 1)

    option = label_one_shot.mul(option_prob) + 1 - (1 - label_one_shot).mul(option_prob)
    option_prob_log = - torch.log(option)
    loss = torch.mean(option_prob_log)
    
    return loss_cross #+ loss
'''

def DSQA_loss(final_result, passage_prob, labels):
    '''
    label_one_shot = torch.zeros(final_result.size()).cuda()
    label_one_shot.scatter_(dim = 1, index = labels.unsqueeze(1), value = 1)

    result = torch.log(final_result + 0.0001)
    loss = label_one_shot.mul(result)
    loss = torch.sum(loss)
    '''
    loss = cross_entropy_loss(final_result, labels)
    # print('loss:', loss)
    
    # print(passage_prob)

    loss_rp = torch.log(passage_prob + 0.0001)
    loss_rp = - 0.005 * torch.sum(loss_rp) / final_result.shape[0]

    # print('loss_rp', loss_rp)
    return loss + loss_rp



def multi_label_cross_entropy_loss(outputs, labels):
    labels = labels.float()
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
