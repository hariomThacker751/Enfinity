import cv2
import numpy as np
import os
from pathlib import Path

mask_dir = r"C:\Users\divya\OneDrive\Desktop\Efinity\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation"
unique_values = set()

files = list(Path(mask_dir).glob("*.png"))[:100] # Check first 100 files
for f in files:
    im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    if im is not None:
        unique_values.update(np.unique(im))

print("Unique values found in masks:", sorted(list(unique_values)))
