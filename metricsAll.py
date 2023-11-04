import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    #验证白色区域
    output_ = output > 0.5
    target_ = target > 0.5
    #验证黑色区域
    output1 = output < 0.5
    target1 = target < 0.5
    #iou
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    #meaniou
    intersection1 = (output1 & target1).sum()
    union1 = (output1 | target1).sum()
    iou1 = (intersection1 + smooth) / (union1 + smooth)
    meanIou = (iou + iou1)/2
    #meandice
    iou2 = (2 * (output_ & target_).sum()) / (output_.sum() + target_.sum())
    iou3 =(2 * (output1 & target1).sum()) / (output1.sum() + target1.sum())
    meandice = (iou2 + iou3) / 2
    #dice
    dice = (2* iou) / (iou+1)
    #acc
    #acc = accuracy_score(output,target)

    # Precision
    precision = intersection / (output_.sum() + smooth)
    # Recall
    recall = intersection / (target_.sum() + smooth)
    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall + smooth)

    return iou, dice, meanIou, f1_score


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
