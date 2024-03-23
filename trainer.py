# Given code is reference from https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/main.py
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding

from myconfig import CFG
from src.data_management.dataset import CxrDatasetVer1, DatasetSplit
from src.models.experiments import CxrVQA
from src.utils import AvgMeter

IS_TESTING = False


def collate_fn(batch):
    imgs = [item[0] for item in batch]

    collated_data = dict()
    for key in batch[0][1].keys():
        data = [item[1][key] for item in batch]
        collated_data[key] = torch.stack(data).squeeze()

    stacked_imgs = torch.stack(imgs)
    stacked_texts = BatchEncoding(data=collated_data)
    return stacked_imgs, stacked_texts


train_ds = CxrDatasetVer1(
    metadata_path="data/metadata/mimic-cxr-2.0.0-metadata.csv",
    split_path="data/metadata/mimic-cxr-2.0.0-split.csv",
    report_dir="data/mimic-cxr-reports",
    img_dir="data/cxr_dataset",
    split=DatasetSplit.TRAIN,
    testing=IS_TESTING,
)

val_ds = CxrDatasetVer1(
    metadata_path="data/metadata/mimic-cxr-2.0.0-metadata.csv",
    split_path="data/metadata/mimic-cxr-2.0.0-split.csv",
    report_dir="data/mimic-cxr-reports",
    img_dir="data/cxr_dataset",
    split=DatasetSplit.VAL,
    testing=IS_TESTING,
)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    collate_fn=collate_fn,
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    collate_fn=collate_fn,
)

model = CxrVQA().to(CFG.device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=CFG.patience,
    factor=CFG.factor,
    min_lr=CFG.min_lr,
)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        imgs = batch[0].to(CFG.device)
        texts = batch[1].to(CFG.device)
        loss = model(imgs, texts)

        count = len(batch[0])
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


best_loss = float("inf")
for epoch in range(CFG.epochs):
    print(f"Epoch: {epoch + 1}")
    model.train()

    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        imgs = batch[0].to(CFG.device)
        texts = batch[1].to(CFG.device)
        loss = model(imgs, texts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step(loss)

        count = len(batch[0])
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(
            train_loss=loss_meter.avg, lr=get_lr(optimizer)
        )

    model.eval()
    with torch.no_grad():
        valid_loss = valid_epoch(model, val_loader)

    if valid_loss.avg < best_loss:
        best_loss = valid_loss.avg
        torch.save(model.state_dict(), "./ckpts/best.pt")
        print("Saved Best Model!")
