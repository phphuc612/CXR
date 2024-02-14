import logging

import torch
import torchxrayvision as xrv
from torch import nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CxrResNet(nn.Module):
    def __init__(self):
        super(CxrResNet, self).__init__()
        self.encoder = xrv.autoencoders.ResNetAE(weights="101-elastic")

    def forward(self, img):
        features = self.encoder.features(img)

        return features
