import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class cohen(nn.Module):

    def __init__(self, n=3):
        super(cohen, self).__init__()
        self.num_classes = n

    def forward(self, preds, targets):
        return torchmetrics.functional.cohen_kappa(preds, targets, self.num_classes, weights=None, threshold=0.5)
