# Enfinity — Off-Road Semantic Segmentation

A semantic segmentation system for off-road environments, built on top of a frozen DINOv2 Vision Transformer backbone with a custom multi-layer decoder head. The model classifies every pixel in an image into one of ten vegetation and terrain categories.

---

## Overview

The system is designed to parse complex, unstructured outdoor scenes into semantic regions. Rather than fine-tuning the entire backbone, only a lightweight segmentation head is trained, which allows the model to generalise from DINOv2's rich pre-trained representations to the off-road domain with relatively few training samples.

**Segmentation classes**

| ID | Class          |
|----|----------------|
| 0  | Trees          |
| 1  | Lush Bushes    |
| 2  | Dry Grass      |
| 3  | Dry Bushes     |
| 4  | Ground Clutter |
| 5  | Flowers        |
| 6  | Logs           |
| 7  | Rocks          |
| 8  | Landscape      |
| 9  | Sky            |

Background pixels are assigned label 255 and are excluded from loss computation and metrics.

---

## Architecture

### Backbone

DINOv2 ViT-L/14 with registers (`dinov2_vitl14_reg`) is used as the feature extractor. The backbone is kept completely frozen during training. Intermediate token outputs from four transformer layers are extracted and reshaped into spatial feature maps.

### Segmentation head (`MultiLayerSegmentationHead`)

Four independent projection branches process each of the four intermediate feature maps (embedding dimension 1024) into a shared hidden dimension of 256 using a Conv2d, GroupNorm, and GELU block. The resulting maps are concatenated along the channel axis and fused by a second convolutional block.

A progressive decoder then upsamples the fused representation through four stages, each doubling the spatial resolution, until the full input resolution is recovered (952 x 532 for 960 x 540 source images). The final layer is a 1x1 convolution that produces per-pixel class logits.

```
Backbone (frozen)
  4 intermediate layers  →  4 x [B, 1024, H/14, W/14]
        │
  Per-layer projection   →  4 x [B, 256, H/14, W/14]
        │
  Channel concatenation  →  [B, 1024, H/14, W/14]
        │
  Fusion conv            →  [B, 256, H/14, W/14]
        │
  Decoder block 1 (x2)  →  [B, 128, H/7, W/7]
  Decoder block 2 (x2)  →  [B, 64,  H/3.5, W/3.5]
  Decoder block 3 (x2)  →  [B, 32,  H/1.75, W/1.75]
  Decoder block 4 (exact pixel size)  →  [B, 16, H, W]
        │
  Head (1x1 conv)        →  [B, 10, H, W]  (logits)
```

---

## Training

### Loss function

Training uses a combined Focal Loss and Dice Loss:

- **Focal Loss** (`alpha=0.25`, `gamma=2.0`) down-weights easy, well-classified pixels so the optimiser concentrates on hard and rare classes such as Logs and Flowers.
- **Dice Loss** directly optimises overlap between predicted and ground-truth regions per class.

Pre-computed log-smoothed inverse-frequency class weights are applied inside the Focal Loss to further counteract class imbalance.

### Optimiser and schedule

- Optimiser: AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- Schedule: 5-epoch linear warm-up from `1e-6` to `1e-4`, followed by cosine decay back to `1e-6` over the remaining epochs.
- Default training length: 45 epochs.

### Gradient accumulation and mixed precision

Effective batch size is 4, achieved through gradient accumulation over 4 steps with a physical batch size of 1. Automatic Mixed Precision (AMP) is used throughout to reduce GPU memory requirements when training at full resolution.

### Data augmentation

Augmentations are applied jointly to image and mask to preserve label alignment:

- Random horizontal flip
- Random rotation (-5 to +5 degrees)
- Random resized crop (scale 0.8 – 1.0, ratio 0.9 – 1.1), applied 50 % of the time
- Colour jitter (brightness, contrast, saturation independently, each 50 %)
- Gaussian blur (sigma 0.1 – 2.0, 50 %)
- Random erasing on image only (30 – 80 px patch, 50 %)

Masks are always resized with nearest-neighbour interpolation to preserve integer class IDs.

### Checkpointing and auto-resume

A checkpoint is saved after every epoch under `checkpoints/epoch_<N>.pt`. The best checkpoint by validation mIoU is also saved as `segmentation_head_best.pt`. Training automatically resumes from the latest epoch checkpoint if one is found.

---

## Evaluation

The validation script computes the following metrics globally across the entire validation set:

- **Mean IoU** (primary metric) — computed as global intersection over global union per class, then averaged across valid classes.
- **Dice coefficient** (mean F1 score across classes)
- **Pixel accuracy**

Per-class IoU values are written to `predictions/evaluation_metrics.txt` and visualised as a bar chart at `predictions/per_class_metrics.png`.

For each processed image the script saves:

- A raw prediction mask containing class IDs 0–9 (`predictions/masks/`)
- A colour-coded RGB visualisation (`predictions/masks_color/`)
- A side-by-side comparison of the input image, ground truth, and prediction for the first N samples (`predictions/comparisons/`)

---

## Dataset structure

The training and validation datasets are expected to follow this layout:

```
Offroad_Segmentation_Training_Dataset/
  train/
    Color_Images/    # RGB images (.png)
    Segmentation/    # Single-channel masks with raw pixel values (.png)
  val/
    Color_Images/
    Segmentation/

Offroad_Segmentation_testImages/
  Color_Images/
  Segmentation/
```

Raw mask pixel values and their corresponding class mappings:

| Raw value | Class          |
|-----------|----------------|
| 0         | Background (ignored) |
| 100       | Trees          |
| 200       | Lush Bushes    |
| 300       | Dry Grass      |
| 500       | Dry Bushes     |
| 550       | Ground Clutter |
| 600       | Flowers        |
| 700       | Logs           |
| 800       | Rocks          |
| 7100      | Landscape      |
| 10000     | Sky            |

---

## Requirements

- Python 3.9+
- PyTorch (CUDA-enabled build recommended)
- torchvision
- numpy
- Pillow
- opencv-python (`cv2`)
- matplotlib
- tqdm

DINOv2 is loaded from a local copy of the `facebookresearch/dinov2` repository placed inside PyTorch's hub cache directory (`~/.cache/torch/hub/facebookresearch_dinov2_main`).

**Windows setup**

Batch scripts for creating and configuring the Conda environment are provided in the `ENV_SETUP/` directory:

```
ENV_SETUP/
  create_env.bat      # Creates the Conda environment
  install_packages.bat  # Installs all required packages
  setup_env.bat       # Combined setup script
```

---

## Usage

### Training

```bash
python train_segmentation.py
```

Training configuration (batch size, learning rate, number of epochs, dataset paths) is set at the top of the `main()` function. Output artefacts are written to `train_stats/` and `checkpoints/`.

### Evaluation and inference

```bash
python test_segmentation.py \
  --model_path segmentation_head_best.pt \
  --data_dir path/to/test/dataset \
  --output_dir predictions \
  --batch_size 1 \
  --num_samples 5
```

If `--model_path` is not provided, the script automatically searches for `segmentation_head_best.pt` or falls back to the latest epoch checkpoint in `checkpoints/`.

### Mask visualisation utility

`visualize.py` is a standalone utility that reads single-channel segmentation masks from a folder and writes colour-coded RGB images to a `colorized/` subfolder. Configure the `input_folder` variable inside the script before running it.

---

## File reference

| File | Description |
|------|-------------|
| `models.py` | `MultiLayerSegmentationHead` model definition |
| `train_segmentation.py` | Full training pipeline including dataset, losses, metrics, and checkpointing |
| `test_segmentation.py` | Validation and inference pipeline with metric reporting and visualisation |
| `visualize.py` | Standalone mask colourisation utility |
| `segmentation_head_best.pt` | Best trained model weights (tracked in git) |
| `ENV_SETUP/` | Windows batch scripts for environment setup |
| `predictions/` | Default output directory for evaluation results |
