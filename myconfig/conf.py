from dataclasses import dataclass

import torch


@dataclass
class CFG:
    batch_size = 96
    num_workers = 8
    lr = 1e-1
    weight_decay = 1e-3
    temperature = 1.0
    epochs = 30

    patience = 1500
    factor = 0.5
    min_lr = 1e-6

    device = "cuda" if torch.cuda.is_available() else "cpu"
