#!/usr/bin/env python3
"""
train.py — Duality AI Offroad Segmentation (V3)
Architecture : UNet + MiT-B2 (Mix Transformer encoder, SCSE decoder attention)
Loss         : 0.6 × Lovász + 0.4 × Focal (γ=2.5)
Scheduler    : CosineAnnealingWarmRestarts (T0=10, Tm=2)
Augmentation : V3 pipeline (geometric + heavy colour/texture)
"""

import subprocess, sys, os, json, glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Configuration  (change ONLY here)
# ──────────────────────────────────────────────────────────────────────────────
DATASET_NAME  = 'duality-offroad-segmentation'
BASE_PATH     = f'/kaggle/input/datasets/rishiikumarsingh/{DATASET_NAME}/data/'
TRAIN_DIR     = os.path.join(BASE_PATH, 'train')
VAL_DIR       = os.path.join(BASE_PATH, 'val')
TEST_DIR      = os.path.join(BASE_PATH, 'test')
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

OUT_DIR       = '/kaggle/working/outputs'
PRED_DIR      = '/kaggle/working/predictions'
CKPT_DIR      = '/kaggle/working/checkpoints'
for d in [OUT_DIR, PRED_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

BEST_PATH     = '/kaggle/working/best_model.pth'
FINAL_PATH    = '/kaggle/working/final_model.pth'

IMAGE_SIZE    = 512
BATCH_SIZE    = 8
NUM_WORKERS   = 2
EPOCHS        = 50
LR            = 2e-4
WEIGHT_DECAY  = 1e-4
COSINE_T0     = 10
COSINE_TMUL   = 2

ENCODER       = 'mit_b2'
ENC_WEIGHTS   = 'imagenet'
NUM_CLASSES   = 10

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass',
    'Dry Bushes', 'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

PALETTE = np.array([
    [0,   0,   0  ],   # Background
    [34,  139, 34 ],   # Trees
    [0,   128, 0  ],   # Lush Bushes
    [210, 180, 140],   # Dry Grass
    [139, 90,  43 ],   # Dry Bushes
    [128, 128, 128],   # Ground Clutter
    [101, 67,  33 ],   # Logs
    [169, 169, 169],   # Rocks
    [210, 105, 30 ],   # Landscape
    [135, 206, 235],   # Sky
], dtype=np.uint8)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
def mask_to_color(mask_np):
    h, w = mask_np.shape
    out  = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask_np == c] = PALETTE[c]
    return out


class SegDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, 'Color_Images')
        self.mask_dir  = os.path.join(root_dir, 'Segmentation')
        self.images    = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name  = self.images[idx]
        img   = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, name)), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
        mask  = np.clip(mask, 0, NUM_CLASSES - 1)
        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        return img, mask.long()

# ──────────────────────────────────────────────────────────────────────────────
# Augmentation — V3 pipeline
# ──────────────────────────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Affine(translate_percent=0.05, scale=(0.85, 1.15), rotate=(-15, 15), p=0.5),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.3),
    A.RandomGamma(gamma_limit=(70, 130), p=0.3),
    A.ChannelShuffle(p=0.1),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ──────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────────────
lovasz_loss = smp.losses.LovaszLoss(mode='multiclass', per_image=False)
focal_loss  = smp.losses.FocalLoss(mode='multiclass', gamma=2.5, normalized=True)

def combined_loss(pred, target):
    """0.6 × Lovász + 0.4 × Focal."""
    return 0.6 * lovasz_loss(pred, target) + 0.4 * focal_loss(pred, target)

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_iou(pred_logits, target, num_classes=NUM_CLASSES):
    pred = torch.argmax(pred_logits, dim=1)
    ious = []
    for c in range(num_classes):
        p = (pred == c); t = (target == c)
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append(float('nan') if union == 0 else (inter / union).item())
    return np.nanmean(ious), ious

def compute_pixel_acc(pred_logits, target):
    return (torch.argmax(pred_logits, dim=1) == target).float().mean().item()

# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def save_checkpoint(epoch, model, optimizer, scheduler, scaler, history, best_iou, tag=''):
    path = os.path.join(CKPT_DIR, f'ckpt_epoch_{epoch:03d}{tag}.pth')
    torch.save({
        'epoch': epoch, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(), 'history': history, 'best_iou': best_iou
    }, path)
    return path

def load_latest_checkpoint(model, optimizer, scheduler, scaler):
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, 'ckpt_epoch_*.pth')))
    if not ckpts:
        return 1, {k: [] for k in ['train_loss','val_loss','train_iou','val_iou','val_acc','lr']}, 0.0
    ckpt = torch.load(ckpts[-1], map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    scaler.load_state_dict(ckpt['scaler'])
    print(f'Resumed from {ckpts[-1]}  (epoch {ckpt["epoch"]})')
    return ckpt['epoch'] + 1, ckpt['history'], ckpt['best_iou']

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f'Device: {DEVICE}')
    if DEVICE == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # DataLoaders
    train_ds = SegDataset(TRAIN_DIR, train_transform)
    val_ds   = SegDataset(VAL_DIR,   val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    # Model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENC_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
        decoder_attention_type='scse',
    ).to(DEVICE)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    # Optimizer — differential LR
    param_groups = [
        {'params': model.encoder.parameters(),           'lr': LR * 0.1},
        {'params': model.decoder.parameters(),           'lr': LR},
        {'params': model.segmentation_head.parameters(), 'lr': LR},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=COSINE_TMUL, eta_min=1e-6)
    scaler    = GradScaler()

    START_EPOCH, history, best_iou = load_latest_checkpoint(model, optimizer, scheduler, scaler)

    # Training loop
    for epoch in range(START_EPOCH, EPOCHS + 1):
        model.train()
        tr_losses, tr_ious = [], []
        for imgs, masks in tqdm(train_loader, desc=f'Ep {epoch:02d}/{EPOCHS} [Train]', leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                out  = model(imgs)
                loss = combined_loss(out, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            tr_losses.append(loss.item())
            miou, _ = compute_iou(out.detach(), masks)
            tr_ious.append(miou)

        scheduler.step(epoch)

        # Validation
        model.eval()
        val_losses, val_ious, val_accs = [], [], []
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f'Ep {epoch:02d}/{EPOCHS} [Val]', leave=False):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with autocast():
                    out  = model(imgs)
                    loss = combined_loss(out, masks)
                val_losses.append(loss.item())
                miou, _ = compute_iou(out, masks)
                val_ious.append(miou)
                val_accs.append(compute_pixel_acc(out, masks))

        tr_l, val_l = np.mean(tr_losses), np.mean(val_losses)
        tr_i, val_i = np.mean(tr_ious),  np.mean(val_ious)
        val_a        = np.mean(val_accs)
        cur_lr       = optimizer.param_groups[-1]['lr']

        for k, v in zip(['train_loss','val_loss','train_iou','val_iou','val_acc','lr'],
                         [tr_l, val_l, tr_i, val_i, val_a, cur_lr]):
            history[k].append(v)

        print(f'Ep {epoch:02d}/{EPOCHS} | '
              f'TrL={tr_l:.4f} VL={val_l:.4f} | '
              f'TrIoU={tr_i:.4f} VIoU={val_i:.4f} | '
              f'Acc={val_a:.4f} | LR={cur_lr:.2e}')

        if val_i > best_iou:
            best_iou = val_i
            torch.save(model.state_dict(), BEST_PATH)
            print(f'  ✅ New best IoU: {best_iou:.4f}')

        if epoch % 10 == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, history, best_iou)

    torch.save(model.state_dict(), FINAL_PATH)
    with open(os.path.join(OUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\nTraining complete. Best Val IoU: {best_iou:.4f}')


if __name__ == '__main__':
    main()
