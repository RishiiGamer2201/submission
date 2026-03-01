#!/usr/bin/env python3
"""
test.py — Duality AI Offroad Segmentation (V3)
Inference with 3-way Test-Time Augmentation (original, H-flip, V-flip).
Loads best_model.pth by default; falls back to latest checkpoint or final_model.pth.
"""

import os, glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_PATH    = '/kaggle/input/datasets/rishiikumarsingh/duality-offroad-segmentation/data/'
TEST_DIR     = os.path.join(BASE_PATH, 'test')
OUT_DIR      = '/kaggle/working/outputs'
PRED_DIR     = '/kaggle/working/predictions'
CKPT_DIR     = '/kaggle/working/checkpoints'
BEST_PATH    = '/kaggle/working/best_model.pth'
FINAL_PATH   = '/kaggle/working/final_model.pth'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_SIZE   = 512
BATCH_SIZE   = 4
NUM_WORKERS  = 2
ENCODER      = 'mit_b2'
NUM_CLASSES  = 10

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass',
    'Dry Bushes', 'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

PALETTE = np.array([
    [0,   0,   0  ],
    [34,  139, 34 ],
    [0,   128, 0  ],
    [210, 180, 140],
    [139, 90,  43 ],
    [128, 128, 128],
    [101, 67,  33 ],
    [169, 169, 169],
    [210, 105, 30 ],
    [135, 206, 235],
], dtype=np.uint8)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def mask_to_color(mask_np):
    h, w = mask_np.shape
    out  = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask_np == c] = PALETTE[c]
    return out


class SegDataset(torch.utils.data.Dataset):
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
        return img, mask.long(), name

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_iou_np(pred, target, num_classes=NUM_CLASSES):
    ious = []
    for c in range(num_classes):
        p = (pred == c); t = (target == c)
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        ious.append(float('nan') if union == 0 else inter / union)
    return np.nanmean(ious), ious

# ──────────────────────────────────────────────────────────────────────────────
# 3-way TTA inference
# ──────────────────────────────────────────────────────────────────────────────
def tta_predict(model, imgs):
    """Average logits over: original, horizontal flip, vertical flip."""
    with autocast():
        p0 = model(imgs)
        p1 = model(torch.flip(imgs, [3]))
        p1 = torch.flip(p1, [3])
        p2 = model(torch.flip(imgs, [2]))
        p2 = torch.flip(p2, [2])
    return (p0 + p1 + p2) / 3.0

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def pick_model():
    if os.path.exists(BEST_PATH):
        return BEST_PATH, 'best checkpoint (highest Val IoU)'
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, '*.pth')))
    if ckpts:
        return ckpts[-1], 'latest periodic checkpoint'
    return FINAL_PATH, 'final model (epoch 50)'


def main():
    print(f'Device: {DEVICE}')

    # Load model
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
        decoder_attention_type='scse',
    ).to(DEVICE)

    load_path, reason = pick_model()
    ckpt = torch.load(load_path, map_location=DEVICE)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    print(f'Loaded: {load_path}  ({reason})')
    model.eval()

    # Dataset
    test_ds = SegDataset(TEST_DIR, val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f'Test images: {len(test_ds)}')

    all_preds, all_gts = [], []

    with torch.no_grad():
        for imgs, masks, names in tqdm(test_loader, desc='Testing (3-way TTA)'):
            imgs  = imgs.to(DEVICE)
            logits = tta_predict(model, imgs)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            gts    = masks.numpy()

            for pred, gt, name in zip(preds, gts, names):
                all_preds.append(pred)
                all_gts.append(gt)
                # Save colour overlay
                color_pred = mask_to_color(pred)
                cv2.imwrite(os.path.join(PRED_DIR, name), cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR))

    # Compute metrics
    all_preds_flat = np.concatenate([p.ravel() for p in all_preds])
    all_gts_flat   = np.concatenate([g.ravel() for g in all_gts])

    class_ious = []
    for c in range(NUM_CLASSES):
        p = (all_preds_flat == c); t = (all_gts_flat == c)
        inter = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        iou   = float('nan') if union == 0 else inter / union
        class_ious.append(iou)

    mean_iou = np.nanmean(class_ious)

    print(f'\nTEST RESULTS (UNet + MiT-B2, V3, 3-way TTA)')
    print(f'Mean IoU: {mean_iou:.4f}\n')
    lines = [f'TEST RESULTS (UNet + MiT-B2, V3, 3-way TTA)\nMean IoU: {mean_iou:.4f}\n']
    for name, iou in zip(CLASS_NAMES, class_ious):
        val = f'{iou:.4f}' if not np.isnan(iou) else 'N/A'
        line = f'  {name:<20}: {val}'
        print(line)
        lines.append(line)

    with open(os.path.join(OUT_DIR, 'test_metrics.txt'), 'w') as f:
        f.write('\n'.join(lines))

    print(f'\nPredictions saved to: {PRED_DIR}')
    print(f'Metrics saved to:     {OUT_DIR}/test_metrics.txt')


if __name__ == '__main__':
    main()
