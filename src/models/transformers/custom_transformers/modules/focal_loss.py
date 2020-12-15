import torch
from torch import nn
from torch.nn import NLLLoss


class FocalLoss(nn.Module):
    """Focal Loss - https://arxiv.org/pdf/1708.02002.pdf"""

    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        loss_fct = NLLLoss(reduction='none')
        CE_loss = loss_fct(inputs, targets)

        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return F_loss.mean()
