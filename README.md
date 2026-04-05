# DINOv2 Off-Road Semantic Segmentation

This repository contains a semantic segmentation pipeline designed to detect off-road terrain classes using a frozen Meta **DINOv2-Large (`vitl14_reg`)** backbone combined with a custom Progressive Semantic Decoder head.

## Key Features

- **Zero-Shot Feature Extraction**: Leverages a frozen DINOv2-Large backbone to prevent overfitting on synthetic off-road datasets.
- **Global Confusion Matrix IoU**: Evaluates exact pixel matches across the entire dataset rather than image-wise macro-averaging, producing more reliable metrics.
- **Weighted Focal Loss**: Addresses class imbalance by targeting rare classes (Logs, Flowers, Rocks) using pre-computed frequency smoothing.
- **DataParallel Compatibility**: Built-in logic strips `DataParallel` and full-model wrappers automatically, enabling direct use of Kaggle checkpoints without manual weight surgery.

---

## 1. Environment and Dependency Requirements

**Prerequisites:** Python 3.10 or later and a CUDA-capable GPU.

Install the required packages:

```bash
pip install torch torchvision
pip install opencv-python matplotlib tqdm Pillow numpy
```

> **Optional (recommended):** Install `xformers` for memory-efficient attention in DINOv2.

---

## 2. Reproducing Results and Training the Model

1. Place the `Offroad_Segmentation_Training_Dataset` directory one level above the script directory.
2. Ensure at least 16 GB of GPU VRAM is available. Alternatively, use the provided Kaggle notebook (`generate_kaggle_nb.py`), which targets dual T4 GPUs.
3. Run the training script:

```bash
python train_segmentation.py
```

**Expected behavior:**
Training runs for 45 epochs by default. Augmentations are performed using native PyTorch tensors to avoid memory issues associated with PIL array interfaces on Windows. IoU and loss metrics are logged per batch, and the best checkpoint is saved automatically as `segmentation_head_best.pt`.

Since only the lightweight MLP classification head is trained while the large backbone remains frozen, training typically completes within a few hours.

---

## 3. Running Inference and Testing

1. Place the model weights file (e.g., `segmentation_head_best.pt`) in the root directory.
2. Run the test script. By default, it reads from `../Offroad_Segmentation_testImages`:

```bash
python test_segmentation.py
```

**Custom paths:**
To use a different checkpoint or image directory, pass the relevant arguments:

```bash
python test_segmentation.py --model_path checkpoints/epoch_25_valiou_0.5489.pt --data_dir /path/to/images
```

---

## 4. Expected Outputs and Interpretation

Running `test_segmentation.py` processes all images and populates a `predictions/` directory with the following contents:

- `masks/`: Raw 8-bit PNG files where each pixel value corresponds to a class ID (0 to 9).
- `masks_color/`: Colorized segmentation masks for visual inspection.
- `comparisons/`: Side-by-side comparison images showing the input, ground truth, and prediction for up to 5 samples.
- `evaluation_metrics.txt` and `per_class_metrics.png`: Quantitative results.
  - **Mean IoU** is computed globally across all pixels in the dataset, avoiding the distortion caused by image-wise macro-averaging.
  - The evaluation approach ensures that rare or absent classes in individual images do not artificially deflate overall IoU scores.
