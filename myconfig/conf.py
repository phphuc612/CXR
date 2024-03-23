from dataclasses import dataclass

import torch


@dataclass
class CFG:
    batch_size = 96
    num_workers = 8
    lr = 1e-1
    weight_decay = 1e-3
    temperature = 0.8
    epochs = 15

    patience = 100
    factor = 0.9
    min_lr = 1e-8

    device = "cuda" if torch.cuda.is_available() else "cpu"
