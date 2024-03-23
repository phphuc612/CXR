# Given code is reference from https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/main.py
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding

from myconfig import CFG
from src.data_management.dataset import (
    CxrDatasetVer1,
    CxrEvaluationCLIP,
    DatasetSplit,
)
from src.models.experiments import CxrVQA
from src.utils import AvgMeter

IS_TESTING = False


def eval_collate_fn(batch):
    imgs = [item[0] for item in batch]

    collated_data = dict()
    for key in batch[0][1].keys():
        data = [item[1][key] for item in batch]
        collated_data[key] = torch.stack(data).squeeze()

    stacked_imgs = torch.stack(imgs)
    stacked_texts = BatchEncoding(data=collated_data)
    return stacked_imgs, stacked_texts


val_ds = CxrEvaluationCLIP(
    label_path="/mnt/ssd1/CXR/data/classification-labels/train.csv",
    img_dir="/mnt/ssd1/CXR/data/cxr_dataset",
    testing=IS_TESTING,
)

val_dataloader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    collate_fn=eval_collate_fn,
)

state_dict = torch.load("./ckpts/vqa_resnet101_clip_ver1.pt")
model = CxrVQA()
model.load_state_dict(state_dict)
model.to(CFG.device)

n_classes = 26

avg_auc = 0


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


model.eval()

for epoch in range(CFG.epochs):
    print(f"Epoch: {epoch + 1}")

    tqdm_object = tqdm(val_dataloader, total=len(val_dataloader))
    for batch in tqdm_object:
        imgs = batch[0].to(CFG.device)
        texts = batch[1].to(CFG.device)
        sim = model.calcluate_similarity(imgs, texts)

        count = len(batch[0])

    model.eval()
    with torch.no_grad():
        valid_loss = valid_epoch(model, val_dataloader)
        print(f"Valid Loss: {valid_loss}")
