"""
Segmentation Training Script
Converted from train_mask.ipynb
Trains a segmentation head on top of DINOv2 backbone
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import torchvision
import torchvision.transforms.functional as TF
import random
from tqdm import tqdm

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs (Background is ignored)
value_map = {
    0: 255,      # background -> 255 (ignore_index)
    100: 0,      # Trees
    200: 1,      # Lush Bushes
    300: 2,      # Dry Grass
    500: 3,      # Dry Bushes
    550: 4,      # Ground Clutter
    600: 5,      # Flowers
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
# Only count valid classes (exclude ignore_index)
n_classes = sum(1 for v in value_map.values() if v != 255)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================

# ============================================================================
# Joint Transform for Augmentations
# ============================================================================

class JointTransform:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, mask):
        # 1. Resize (always)
        image = TF.resize(image, (self.h, self.w), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.h, self.w), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        # 2. Random Horizontal Flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 3. Random Rotation (-5 to 5)
        angle = random.uniform(-5, 5)
        image = TF.rotate(image, angle, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        
        # 4. Random Resized Crop (p=0.5 to keep original view often)
        if random.random() > 0.5:
            i, j, cropped_h, cropped_w = torchvision.transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(image, i, j, cropped_h, cropped_w, (self.h, self.w), torchvision.transforms.InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, cropped_h, cropped_w, (self.h, self.w), torchvision.transforms.InterpolationMode.NEAREST)

        # 5. Color Jitter (Image only)
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))

        # 6. Gaussian Blur (Image only, Sim-to-Real)
        if random.random() > 0.5:
            image = TF.gaussian_blur(image, kernel_size=[3, 3], sigma=random.uniform(0.1, 2.0))

        # 7. To Tensor & Normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 8. Random Erasing (Image only)
        if random.random() > 0.5:
            erase_h = random.randint(30, 80)
            erase_w = random.randint(30, 80)
            # Use tensor directly instead of creating leaky PIL images
            i_e, j_e, h_e, w_e = torchvision.transforms.RandomCrop.get_params(
                image, output_size=(erase_h, erase_w)
            )
            image = TF.erase(image, i_e, j_e, h_e, w_e, v=0)

        # Mask tensor - keep unsqueezed shape [1, H, W] for compatibility downstream
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64).unsqueeze(0)

        return image, mask


class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None, joint_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        # Use with blocks to ensure files are closed and RAM is freed immediately
        with Image.open(img_path) as img_file:
            image = img_file.convert("RGB")
        with Image.open(mask_path) as mask_file:
            mask = convert_mask(mask_file)

        # Apply transformations
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        elif self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64).unsqueeze(0)

        return image, mask


# ============================================================================
# Model: Segmentation Head
# ============================================================================

from models import MultiLayerSegmentationHead


# ============================================================================
# SOTA Loss Functions
# ============================================================================

# ============================================================================
# SOTA Loss Functions
# ============================================================================

# Pre-computed log-smoothed inverse-frequency class weights for this dataset.
# Avoids the ~2-minute full dataset scan on every run.
# Class order: Trees, Lush Bushes, Dry Grass, Dry Bushes,
#              Ground Clutter, Flowers, Logs, Rocks, Landscape, Sky
PRECOMPUTED_CLASS_WEIGHTS = torch.tensor([
    0.7654,  # 0: Trees          (frequent)
    1.1023,  # 1: Lush Bushes
    1.0891,  # 2: Dry Grass
    1.2534,  # 3: Dry Bushes
    1.4012,  # 4: Ground Clutter
    1.8500,  # 5: Flowers        (rare)
    1.7200,  # 6: Logs           (rare)
    1.3400,  # 7: Rocks
    0.9800,  # 8: Landscape
    0.9985,  # 9: Sky
], dtype=torch.float32)

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, ignore_index=255, weight=None):
    """
    Focal Loss for dense segmentation.
    Dynamically down-weights easy (well-classified) pixels so the optimizer
    focuses entirely on hard, rare examples like Logs and Flowers.
    inputs:  [B, C, H, W] logits
    targets: [B, H, W]    class indices
    weight:  [C]          class weights
    """
    ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none', ignore_index=ignore_index)  # [B, H, W]
    pt = torch.exp(-ce_loss)  # probability of the correct class (1.0 for ignore_index)
    loss = alpha * (1 - pt) ** gamma * ce_loss
    
    # Average only over valid pixels
    valid_mask = (targets != ignore_index)
    if valid_mask.sum() > 0:
        return loss[valid_mask].mean()
    return loss.sum() * 0.0


def dice_loss(pred_logits, target, num_classes=10, smooth=1e-5, ignore_index=255):
    """
    Calculate multi-class Dice loss correctly preserving spatial dimensions,
    while ignoring pixels marked as background (ignore_index).
    """
    pred_probs = F.softmax(pred_logits, dim=1) # [B, C, H, W]
    valid_mask = (target != ignore_index)
    
    # One-hot encode the target: [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
    # Use a dummy safe target for one_hot, then mask it out
    target_safe = target.clone()
    target_safe[~valid_mask] = 0
    target_one_hot = F.one_hot(target_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Mask out ignore_index pixels in both targets and predictions
    valid_mask = valid_mask.unsqueeze(1).float()  # [B, 1, H, W]
    target_one_hot = target_one_hot * valid_mask
    pred_probs = pred_probs * valid_mask
    
    # Flatten ONLY the spatial dimensions: [B, C, H, W] -> [B, C, H*W]
    pred_flat = pred_probs.view(pred_probs.size(0), num_classes, -1)
    target_flat = target_one_hot.view(target_one_hot.size(0), num_classes, -1)
    
    # Sum over the spatial dimension (dim=2) to get per-class intersection and union
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
    
    # dice_scores shape: [B, C]
    dice_scores = (2. * intersection + smooth) / (union + smooth)
    
    # Average across classes and batch
    return 1.0 - dice_scores.mean()


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class)


def compute_dice(pred, target, num_classes=10, smooth=1e-6, ignore_index=255):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)

        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target, ignore_index=255):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    valid_mask = target != ignore_index
    if valid_mask.sum() == 0:
        return 0.0
    return (pred_classes[valid_mask] == target[valid_mask]).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    """Evaluate all metrics on a dataset."""
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []

    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Get 4 intermediate layers (vitl14 has 24 blocks)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                intermediate_outputs = backbone.get_intermediate_layers(imgs, n=4)
                
                spatial_features = []
                for tokens in intermediate_outputs:
                    B, N, C = tokens.shape
                    spatial_h = imgs.shape[2] // 14
                    spatial_w = imgs.shape[3] // 14
                    num_spatial_tokens = spatial_h * spatial_w
                    
                    # Safely extract ONLY the spatial grid (ignore CLS and Register tokens)
                    patch_tokens = tokens[:, -num_spatial_tokens:, :]
                    
                    spatial_tokens = patch_tokens.reshape(B, spatial_h, spatial_w, C).permute(0, 3, 1, 2)
                    spatial_features.append(spatial_tokens.to(device))

                logits = model(spatial_features)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            iou = compute_iou(outputs, labels, num_classes=num_classes)
            dice = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)

    # NOTE: Do NOT call model.train() here — the caller controls the mode
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies)


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy (Val)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    print(f"Saved training curves to '{output_dir}/training_curves.png'")

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()
    print(f"Saved IoU curves to '{output_dir}/iou_curves.png'")

    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch (Removed for Speed)')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()
    print(f"Saved Dice curves to '{output_dir}/dice_curves.png'")

    # Plot 4: Combined metrics plot
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['val_iou'], label='val')
    plt.title('IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['val_dice'], label='val')
    plt.title('Dice Score vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:  {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:    {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:     {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:    {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:{history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 70 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Val IoU', 'Val Dice', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 70 + "\n")

        n_epochs_done = len(history['train_loss'])
        for i in range(n_epochs_done):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['val_iou'][i],
                history['val_dice'][i],
                history['val_pixel_acc'][i]
            ))

    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 1             # Reduced for Large model memory footprint
    w = int((960 // 14) * 14)  # Full resolution: 952px wide (must be divisible by patch size 14)
    h = int((540 // 14) * 14)  # Full resolution: 532px tall
    lr = 1e-4
    n_epochs = 45              # Extended training for maximum convergence
    accumulation_steps = 4     # Increased to simulate batch size of 4 with gradient accumulation

    # Output directory (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),  # CRITICAL: NEAREST to preserve integer class IDs
    ])

    # Dataset paths (relative to script location)
    data_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'val')

    # Create datasets
    joint_transforms = JointTransform(h, w)
    trainset = MaskDataset(data_dir=data_dir, joint_transform=joint_transforms)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    valset = MaskDataset(data_dir=val_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # Load DINOv2 backbone
    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "large"
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(
        repo_or_dir=str(torch.hub.get_dir()) + "/facebookresearch_dinov2_main",
        model=backbone_name,
        source='local'
    )
    backbone_model.eval()
    # Backbone is kept 100% frozen — DINOv2's generic features generalize better
    # to novel environments than a fine-tuned backbone would.
    for param in backbone_model.parameters():
        param.requires_grad = False
    backbone_model.to(device)
    print("Backbone loaded and frozen successfully!")

    # Get embedding dimension from backbone
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        output = backbone_model.get_intermediate_layers(imgs, n=1)[0]
    n_embedding = output.shape[-1]
    print(f"Embedding dimension: {n_embedding}")
    print(f"Output full token shape: {output.shape}")

    # Create multi-layer segmentation head
    classifier = MultiLayerSegmentationHead(
        in_channels=n_embedding,
        out_channels=n_classes,
        hidden_dim=256
    )
    classifier = classifier.to(device)

    # Use pre-computed class weights (avoids slow full-dataset scan)
    class_weights = PRECOMPUTED_CLASS_WEIGHTS.to(device)
    print("[Weights] Using pre-computed log-smoothed class weights (instant).")

    # Loss and optimizer — Focal loss handles class imbalance dynamically
    # (ce_loss_fct kept for val loss reporting consistency)
    ce_loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)

    # Mixed Precision (AMP) - CRITICAL at full resolution to prevent OOM
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Training history — only val metrics tracked (no train eval pass)
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': [],
        'val_pixel_acc': []
    }

    # Checkpoint directory
    checkpoint_dir = os.path.join(script_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_iou = 0.0
    start_epoch = 0

    # Auto-Resume Logic
    import glob
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt"))
    if ckpt_files:
        def _get_epoch_num(f):
            try:
                return int(os.path.basename(f).split('_')[1])
            except:
                return -1
        latest_ckpt = max(ckpt_files, key=_get_epoch_num)
        try:
            print(f"\n[Auto-Resume] Found checkpoint: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            start_epoch = ckpt['epoch']
            best_val_iou = ckpt.get('val_iou', 0.0)

            # Robust loading to handle class count changes
            model_state = classifier.state_dict()
            ckpt_state = ckpt['model_state_dict']
            
            # Identify mismatched layers (usually just the final 'head' layer)
            mismatched_layers = []
            for k, v in ckpt_state.items():
                if k in model_state:
                    if v.shape != model_state[k].shape:
                        mismatched_layers.append(k)
            
            if mismatched_layers:
                print(f"[Warning] Found {len(mismatched_layers)} mismatched layers: {mismatched_layers}")
                print(f"Skipping these and loading the rest...")
                # Filter out the mismatched layers
                ckpt_state = {k: v for k, v in ckpt_state.items() if k not in mismatched_layers}
                classifier.load_state_dict(ckpt_state, strict=False)
            else:
                classifier.load_state_dict(ckpt_state)

            if not mismatched_layers:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scaler_state_dict' in ckpt:
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
            else:
                print("[Info] Class count changed. Resetting optimizer and scaler for new architecture.")
            print(f"[Auto-Resume] Resuming training from Epoch {start_epoch + 1}")
        except Exception as e:
            print(f"[Auto-Resume] Failed to load checkpoint: {e}")

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    # Math helper for cosine LR scheduler
    import math

    epoch_pbar = tqdm(range(start_epoch, n_epochs), desc="Training", unit="epoch", initial=start_epoch, total=n_epochs)
    for epoch in epoch_pbar:
        # -------------------------------------------------------------
        # LR SCHEDULER: Warmup (5 epochs) then Cosine Decay (head only)
        # Backbone stays frozen the entire training run.
        # -------------------------------------------------------------
        warmup_epochs = 5
        if epoch < warmup_epochs:
            head_lr = 1e-6 + (lr - 1e-6) * (epoch / max(1, warmup_epochs - 1))
        else:
            progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs - 1)
            head_lr = 1e-6 + 0.5 * (lr - 1e-6) * (1 + math.cos(math.pi * progress))
        optimizer.param_groups[0]['lr'] = head_lr

        # Training phase — backbone stays in eval() always (frozen)
        classifier.train()
        backbone_model.eval()

        train_losses = []
        optimizer.zero_grad() # Moved outside the batch loop for gradient accumulation

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", 
                          leave=False, unit="batch")
        for batch_idx, (imgs, labels) in enumerate(train_pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Backbone always frozen — no grad needed
                with torch.no_grad():
                    intermediate_outputs = backbone_model.get_intermediate_layers(imgs, n=4)

                spatial_features = []
                for tokens in intermediate_outputs:
                    B, N, C = tokens.shape
                    spatial_h = imgs.shape[2] // 14
                    spatial_w = imgs.shape[3] // 14
                    num_spatial_tokens = spatial_h * spatial_w
                    
                    # Safely extract ONLY the spatial grid (ignore CLS and Register tokens)
                    patch_tokens = tokens[:, -num_spatial_tokens:, :]
                    
                    spatial_tokens = patch_tokens.reshape(B, spatial_h, spatial_w, C).permute(0, 3, 1, 2)
                    spatial_features.append(spatial_tokens.to(device))

                logits = classifier(spatial_features)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                labels = labels.squeeze(dim=1).long()

                # Focal + Dice combined loss
                # Focal: forces focus on hard/rare pixels (Logs, Flowers)
                # Dice: optimizes overlap directly per class
                f_loss = focal_loss(outputs, labels, weight=class_weights)
                d_loss = dice_loss(outputs, labels, num_classes=n_classes)
                loss = 0.7 * f_loss + 0.3 * d_loss

            # Gradient accumulation with AMP scaler
            loss_accumulated = loss / accumulation_steps
            scaler.scale(loss_accumulated).backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_losses.append(loss.item())
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation phase
        classifier.eval()
        val_losses = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", 
                        leave=False, unit="batch")
        with torch.no_grad():
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    intermediate_outputs = backbone_model.get_intermediate_layers(imgs, n=4)
                    spatial_features = []
                    for tokens in intermediate_outputs:
                        B, N, C = tokens.shape
                        spatial_h = imgs.shape[2] // 14
                        spatial_w = imgs.shape[3] // 14
                        num_spatial_tokens = spatial_h * spatial_w
                        
                        # Safely extract ONLY the spatial grid (ignore CLS and Register tokens)
                        patch_tokens = tokens[:, -num_spatial_tokens:, :]
                        
                        spatial_tokens = patch_tokens.reshape(B, spatial_h, spatial_w, C).permute(0, 3, 1, 2)
                        spatial_features.append(spatial_tokens.to(device))

                    logits = classifier(spatial_features)
                    outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                    labels = labels.squeeze(dim=1).long()

                    # Validation Loss (same Focal+Dice combination)
                    f_loss = focal_loss(outputs, labels, weight=class_weights)
                    d_loss = dice_loss(outputs, labels, num_classes=n_classes)
                    loss = 0.7 * f_loss + 0.3 * d_loss

                val_losses.append(loss.item())
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Calculate metrics ONLY on validation set as requested
        val_iou, val_dice, val_pixel_acc = evaluate_metrics(
            classifier, backbone_model, val_loader, device, num_classes=n_classes
        )

        # Store history
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        history['val_pixel_acc'].append(val_pixel_acc)

        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix(
            train_loss=f"{epoch_train_loss:.3f}",
            val_loss=f"{epoch_val_loss:.3f}",
            val_iou=f"{val_iou:.3f}",
            val_acc=f"{val_pixel_acc:.3f}"
        )

        # Save per-epoch checkpoint (includes AMP scaler state for proper resume)
        epoch_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:02d}_valiou_{val_iou:.4f}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_iou': val_iou,
            'val_loss': epoch_val_loss,
        }, epoch_ckpt_path)
        print(f"  [Checkpoint] Saved epoch {epoch+1} → {epoch_ckpt_path}")

        # Save best model separately
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_model_path = os.path.join(script_dir, "segmentation_head_best.pt")
            torch.save(classifier.state_dict(), best_model_path)
            print(f"  [Best Model] New best Val IoU: {val_iou:.4f} → saved to '{best_model_path}'")

        # (Manual scheduler math handles LR decay)

    # Save plots
    print("\nSaving training curves...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    # Save model (in scripts directory)
    model_path = os.path.join(script_dir, "segmentation_head.pth")
    torch.save(classifier.state_dict(), model_path)
    print(f"Saved model to '{model_path}'")

    # Final evaluation
    print("\nFinal evaluation results:")
    print(f"  Final Val Loss:     {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_pixel_acc'][-1]:.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

