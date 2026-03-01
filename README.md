# Duality AI – Offroad Terrain Segmentation
## Team Submission · UNet + MiT-B2 · V3 Pipeline

---

## Results Summary

| Metric | Value |
|--------|-------|
| **Mean IoU (Test)** | **0.3309** |
| Best Val IoU (training) | 0.5202 |
| Val Pixel Accuracy (ep 50) | 87.2% |
| TTA | 3-way (orig + H-flip + V-flip) |

**Per-class IoU:**

| Class | IoU |
|-------|-----|
| Sky | 0.979 |
| Landscape | 0.668 |
| Dry Grass | 0.459 |
| Trees | 0.409 |
| Dry Bushes | 0.362 |
| Rocks | 0.075 |
| Lush Bushes | 0.005 |
| Background | 0.000 |
| Ground Clutter | 0.000 |
| Logs | 0.000 |

---

## Repository Structure

```
submission/
├── train.py                  # Full training script
├── test.py                   # Inference + 3-way TTA
├── duality-v3.ipynb          # Original Kaggle notebook (all iterations)
├── README.md                 # This file
├── report/
│   └── hackathon_report.docx # Full written report
└── assets/
    ├── training_curves.png
    ├── per_class_iou.png
    ├── augmentation_preview.png
    └── test_metrics.txt
```

---

## Environment & Dependencies

### Hardware
- GPU: NVIDIA T4 / P100 (Kaggle)
- RAM: 13 GB

### Python packages

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install segmentation-models-pytorch albumentations timm tqdm opencv-python matplotlib
```

Or use the Kaggle environment directly (all packages pre-installed).

---

## How to Reproduce Results

### 1 · Dataset Setup

The dataset must be available at:
```
/kaggle/input/datasets/rishiikumarsingh/duality-offroad-segmentation/data/
├── train/
│   ├── Color_Images/   # .png RGB images
│   └── Segmentation/   # .png single-channel class masks (0–9)
├── val/
└── test/
```

> ⚠️ Test images were **never** used for training or validation.

### 2 · Train

```bash
python train.py
```

Key hyperparameters (edit at top of `train.py`):

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 512 |
| BATCH_SIZE | 8 |
| EPOCHS | 50 |
| LR | 2e-4 |
| Encoder | mit_b2 (ImageNet weights) |
| Loss | 0.6×Lovász + 0.4×Focal (γ=2.5) |
| Scheduler | CosineAnnealingWarmRestarts (T0=10, Tm=2) |

Checkpoints are saved every 10 epochs to `/kaggle/working/checkpoints/`.  
The best checkpoint (by Val IoU) is saved as `best_model.pth`.

**Auto-resume:** Re-running `train.py` automatically resumes from the latest checkpoint.

### 3 · Test / Inference

```bash
python test.py
```

- Loads `best_model.pth` automatically (falls back to latest checkpoint or `final_model.pth`)
- Applies **3-way TTA**: original + horizontal flip + vertical flip
- Saves colour-coded prediction PNGs to `/kaggle/working/predictions/`
- Writes `test_metrics.txt` to `/kaggle/working/outputs/`

### 4 · Expected Outputs

```
/kaggle/working/
├── best_model.pth
├── final_model.pth
├── checkpoints/
│   ├── ckpt_epoch_010.pth
│   ├── ckpt_epoch_020.pth
│   └── ...
├── outputs/
│   ├── test_metrics.txt
│   └── training_history.json
└── predictions/
    └── *.png   # colour-coded segmentation maps
```

---

## Model Architecture

```
UNet
├── Encoder : MiT-B2 (Mix Transformer, ImageNet pretrained, ~25M params)
│             Differential LR — encoder: LR×0.1, decoder: LR×1.0
└── Decoder : UNet decoder with SCSE attention (Squeeze-Channel-Spatial Excitation)
              Classes: 10
```

---

## Approaches Tried (Chronological)

| Version | Architecture | Key Change | Val IoU |
|---------|-------------|------------|---------|
| V1 | UNet + ResNet50 | Baseline | 0.419 |
| SAM experiment | SAM ViT-H | Zero-shot segmentation | ~0.15 (poor) |
| DINOv2 experiment | DINOv2 + linear probe | Feature extraction | ~0.22 |
| V2 | UNet + MiT-B2 | OneCycleLR (bugged) | 0.277 |
| **V3** | **UNet + MiT-B2** | **CosineWarmRestarts + V3 aug** | **0.5202** |

See `duality-v3.ipynb` for full code history and cell-by-cell commentary.

---

## Notes on Output Interpretation

- **IoU of 0.000**: class absent or entirely misclassified (Background, Ground Clutter, Logs)
- **Sky (0.979)**: easy to segment due to distinct colour/texture
- **Lush Bushes / Logs**: rare classes — very few pixels in the dataset, model fails to learn
- The gap between Val IoU (~0.52) and Test IoU (~0.33) suggests domain shift or test set class distribution differs from validation

---

## Contact

Submit issues via the hackathon Discord channel.
