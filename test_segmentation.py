"""
Segmentation Validation Script
Converted from val_mask.ipynb
Evaluates a trained segmentation head on validation data and saves predictions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
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
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 255,      # background -> ignored
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

# Class names for visualization
class_names = [
    'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = sum(1 for v in value_map.values() if v != 255)

# Color palette for visualization (10 distinct colors)
color_palette = np.array([
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [255, 105, 180],  # Flowers - hot pink
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        # Both color images and masks are .png files with same name
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.int64).unsqueeze(0)

        return image, mask, data_id


# ============================================================================
# Model: Segmentation Head - Must match training
# ============================================================================

from models import MultiLayerSegmentationHead


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute raw intersection and union counts per class for Global IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    intersections = np.zeros(num_classes)
    unions = np.zeros(num_classes)

    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue

        pred_inds = pred == class_id
        target_inds = target == class_id

        intersections[class_id] = (pred_inds & target_inds).sum().item()
        unions[class_id] = (pred_inds | target_inds).sum().item()

    return intersections, unions


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
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

    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Visualization Functions
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save a side-by-side comparison of input, ground truth, and prediction."""
    # Denormalize image
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # Convert masks to color
    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    """Save metrics summary to a text file and create bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    # Save text summary
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for i, (name, iou) in enumerate(zip(class_names, results['class_iou'])):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {iou_str}\n")

    print(f"\nSaved evaluation metrics to {filepath}")

    # Create bar chart for per-class IoU
    fig, ax = plt.subplots(figsize=(10, 6))

    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    ax.bar(range(n_classes), valid_iou, color=[color_palette[i] / 255 for i in range(n_classes)],
           edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', label='Mean')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")


# ============================================================================
# Main Validation Function
# ============================================================================

def main():
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Auto-detect best available model: prefer segmentation_head_best.pt,
    # then fall back to latest epoch checkpoint in checkpoints/
    import glob
    default_model_path = os.path.join(script_dir, 'segmentation_head.pth')
    if not os.path.exists(default_model_path):
        best_pt = os.path.join(script_dir, 'segmentation_head_best.pt')
        if os.path.exists(best_pt):
            default_model_path = best_pt
        else:
            # Fall back to latest epoch checkpoint
            ckpt_files = glob.glob(os.path.join(script_dir, 'checkpoints', 'epoch_*.pt'))
            if ckpt_files:
                def _epoch_num(f):
                    try: return int(os.path.basename(f).split('_')[1])
                    except: return -1
                default_model_path = max(ckpt_files, key=_epoch_num)
                print(f"[Info] segmentation_head.pth not found. Using checkpoint: {os.path.basename(default_model_path)}")

    parser = argparse.ArgumentParser(description='Segmentation prediction/inference script')
    parser.add_argument('--model_path', type=str, default=default_model_path,
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, default=os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages', 'Offroad_Segmentation_testImages'),
                        help='Path to validation dataset')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save prediction visualizations')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for validation')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of comparison visualizations to save (predictions saved for ALL images)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Image dimensions (must match training)
    w = int((960 // 14) * 14)  # Full resolution: 952px wide
    h = int((540 // 14) * 14)  # Full resolution: 532px tall

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),  # CRITICAL: NEAREST to preserve integer class IDs
    ])

    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    valset = MaskDataset(data_dir=args.data_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(valset)} samples")

    # Load DINOv2 backbone
    print("Loading DINOv2 backbone...")
    BACKBONE_SIZE = "large"  # Reverted to Large model as requested
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
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    # Get embedding dimension
    sample_img, _, _ = valset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = backbone_model.get_intermediate_layers(sample_img, n=1)[0]
    n_embedding = output.shape[-1]
    print(f"Embedding dimension: {n_embedding}")

    # Load classifier — handles both:
    #   1. Plain state dict (.pth saved at end of training)
    #   2. Full checkpoint dict (.pt saved per-epoch, contains model_state_dict key)
    print(f"Loading model from {args.model_path}...")
    classifier = MultiLayerSegmentationHead(
        in_channels=n_embedding,
        out_channels=n_classes,
        hidden_dim=256
    )
    raw_ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    # Unwrap checkpoint dict if needed
    if isinstance(raw_ckpt, dict) and 'model_state_dict' in raw_ckpt:
        print(f"  [Checkpoint] Epoch {raw_ckpt.get('epoch','?')}, Val IoU: {raw_ckpt.get('val_iou', '?')}")
        model_ckpt = raw_ckpt['model_state_dict']
    else:
        model_ckpt = raw_ckpt
    # Safely strip DataParallel or FullModel wrappers from Kaggle checkpoints
    cleaned_ckpt = {}
    for k, v in model_ckpt.items():
        if k.startswith('module.classifier.'):
            cleaned_ckpt[k.replace('module.classifier.', '')] = v
        elif k.startswith('classifier.'):
            cleaned_ckpt[k.replace('classifier.', '')] = v
        elif not k.startswith('module.backbone.') and not k.startswith('backbone.'):
            cleaned_ckpt[k] = v

    model_ckpt = cleaned_ckpt
    model_state = classifier.state_dict()
    
    # Identify mismatched layers (usually just the final 'head' layer)
    mismatched_layers = []
    for k, v in model_ckpt.items():
        if k in model_state:
            if v.shape != model_state[k].shape:
                mismatched_layers.append(k)
    
    if mismatched_layers:
        print(f"[Warning] Found {len(mismatched_layers)} mismatched layers in checkpoint: {mismatched_layers}")
        print(f"Skipping these and loading matching weights only...")
        model_ckpt = {k: v for k, v in model_ckpt.items() if k not in mismatched_layers}
        classifier.load_state_dict(model_ckpt, strict=False)
    else:
        classifier.load_state_dict(model_ckpt, strict=False)
    
    classifier = classifier.to(device)
    classifier.eval()
    print("Model loaded successfully!")

    # Create subdirectories for outputs
    masks_dir = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # Run evaluation and save predictions for ALL images
    print(f"\nRunning evaluation and saving predictions for all {len(valset)} images...")

    total_intersections = np.zeros(n_classes)
    total_unions = np.zeros(n_classes)
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Processing", unit="batch")
        for batch_idx, (imgs, labels, data_ids) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                intermediate_outputs = backbone_model.get_intermediate_layers(imgs, n=4)
                spatial_features = []
                for tokens in intermediate_outputs:
                    B, N, C = tokens.shape
                    spatial_h = imgs.shape[2] // 14
                    spatial_w = imgs.shape[3] // 14
                    num_spatial_tokens = spatial_h * spatial_w
                    patch_tokens = tokens[:, -num_spatial_tokens:, :]
                    spatial_tokens = patch_tokens.reshape(B, spatial_h, spatial_w, C).permute(0, 3, 1, 2)
                    spatial_features.append(spatial_tokens.to(device))

                logits = classifier(spatial_features)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            # If labels has shape [B, 1, H, W] squeeze it. Testing script does not use JointTransform
            if len(labels.shape) == 4:
                labels_squeezed = labels.squeeze(dim=1).long()
            else:
                labels_squeezed = labels.long()

            # Calculate metrics
            intersections, unions = compute_iou(outputs, labels_squeezed, num_classes=n_classes)
            # Store global counts instead of pre-computed ratios
            total_intersections += intersections
            total_unions += unions

            # Save predictions for every image
            predicted_masks = torch.argmax(outputs, dim=1)  # [B, H, W]
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                # Save raw prediction mask (class IDs 0-9)
                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
                pred_img = Image.fromarray(pred_mask)
                pred_img.save(os.path.join(masks_dir, f'{base_name}_pred.png'))

                # Save colored prediction mask (RGB visualization)
                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Save comparison visualization for first N samples
                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_squeezed[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f'sample_{sample_count}_comparison.png'),
                        data_id
                    )

                sample_count += 1

            # Update progress bar
            pbar.update(1)

    # Aggregate Global Results correctly
    avg_class_iou = np.zeros(n_classes)
    for class_id in range(n_classes):
        if total_unions[class_id] > 0:
            avg_class_iou[class_id] = total_intersections[class_id] / total_unions[class_id]
        else:
            avg_class_iou[class_id] = np.nan

    mean_iou = np.nanmean(avg_class_iou)

    results = {
        'mean_iou': mean_iou,
        'class_iou': avg_class_iou
    }

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU:          {mean_iou:.4f}")
    print("-" * 50)
    print("Per-Class IoU:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:<16}: {avg_class_iou[i]:.4f}")
    print("=" * 50)

    # Save all results
    save_metrics_summary(results, args.output_dir)

    print(f"\nPrediction complete! Processed {len(valset)} images.")
    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  - masks/           : Raw prediction masks (class IDs 0-10)")
    print(f"  - masks_color/     : Colored prediction masks (RGB)")
    print(f"  - comparisons/     : Side-by-side comparison images ({args.num_samples} samples)")
    print(f"  - evaluation_metrics.txt")
    print(f"  - per_class_metrics.png")


if __name__ == "__main__":
    main()

