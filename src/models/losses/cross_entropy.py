import torch.nn.functional as F
from torch import nn


class CELoss(nn.Module):
    """Cross Entropy Loss for Image-Text Matching"""

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, reduction=None):
        log_probs = F.log_softmax(preds, dim=-1)
        loss = (-targets * log_probs).sum(dim=-1)

        if reduction == "mean":
            return loss.mean()

        return loss
