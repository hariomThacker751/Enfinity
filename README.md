# DINOv2 Offroad Semantic Segmentation

This repository contains heavily optimized semantic segmentation code designed to detect off-road terrain classes using a frozen Meta **DINOv2-Large (`vitl14_reg`)** backbone combined with a custom Progressive Semantic Decoder head.

## 🚀 Key Features
- **Zero-Shot Feature Extraction**: Leverages frozen DINOv2-Large to prevent overfitting on synthetic off-road datasets.
- **Global Confusion Matrix IoU**: Evaluates exact pixel matches across the entire dataset rather than skewed image-wise macro-averaging.
- **Weighted Focal Loss**: Forcefully targets heavily imbalanced/rare rare classes (Logs, Flowers, Rocks) using pre-computed frequency smoothing techniques.
- **DataParallel Kaggle Patched**: Built-in logic strips `DataParallel` and Full-Model wrappers natively, allowing direct download and testing of Kaggle checkpoints without manual tensor surgery.

---

## 💻 1. Environment & Dependency Requirements
You must have Python 3.10+ and a CUDA-capable GPU. 

Install the required packages using pip:
```bash
pip install torch torchvision
pip install opencv-python matplotlib tqdm Pillow numpy
```
*(Optional but recommended): Install `xformers` for faster memory-efficient attention in DINOv2.*

## 🏗️ 2. How to Reproduce Results & Train the Model
1. Place the `Offroad_Segmentation_Training_Dataset` directory one level above the script directory.
2. Ensure you have at least 16GB of GPU VRAM (or run this via the provided Kaggle automation `.ipynb` script `generate_kaggle_nb.py` which targets dual-T4 GPUs).
3. Execute the training script:
```bash
python train_segmentation.py
```
**Expected Output:** 
The code runs 45 epochs by default. During training, it heavily utilizes native PyTorch tensors for augmentations to prevent Windows RAM leaks from PIL array interfaces. It will log IoU and Loss metrics per batch and automatically save `segmentation_head_best.pt`. 

Because only the small MLP classification head is trained (while the billion-parameter backbone is frozen), training finishes in just a few hours rather than days.

## 🧪 3. Step-by-Step Instructions to Run Testing
Testing is engineered extremely strictly to simulate challenge bounds.

1. Verify your target weights file (e.g. `segmentation_head_best.pt`) is sitting in the root folder.
2. Run the test script. Note that the script automatically targets the default competition folder (`../Offroad_Segmentation_testImages`) so you don't even need to pass variables:
```bash
python test_segmentation.py
```

**Custom Testing:** 
If your valid images are somewhere else or you want to test a downloaded Kaggle dict (like `epoch_25_valiou_0.5489.pt`), simply pass the arguments:
```bash
python test_segmentation.py --model_path checkpoints/epoch_25_valiou_0.5489.pt --data_dir /path/to/images 
```

## 📊 4. Expected Outputs & Interpretation
When you run `test_segmentation.py`, expect the progress bar to parse all images and generate a completely populated `predictions/` folder.

Inside `predictions/`, you will find:
*   `masks/`: Raw 8-bit integer pngs representing class IDs from 0 to 9.
*   `masks_color/`: Vibrant, fully colorized masks mapped perfectly to the test visuals.
*   `comparisons/`: 5 high-resolution matrices putting the Input Image, Ground Truth Object, and Prediction side-by-side for human diagnosis.
*   `evaluation_metrics.txt` / `per_class_metrics.png`:
    *   *Interpretation*: **Mean IoU** represents the mathematically sound intersection of all properly clustered class-pixels across the entire spatial range. 
    *   **Note**: The custom testing script avoids standard batch macro-averaging, meaning tiny hallucinated classes in empty images will no longer artificially tank the IoU into the 0.00% range.
