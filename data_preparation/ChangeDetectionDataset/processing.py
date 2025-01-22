#%%
import os
import pathlib

import cv2
import numpy as np

# image_path = "D:/storage/ChangeDetectionDataset/Real/subset/test/A/00000.jpg"
image_path = "D:/storage/ChangeDetectionDataset/Real/subset/test/OUT/00000.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

pixel_values = np.array(img, dtype=int)

for row in pixel_values:
    print(" ".join(f"{val:3}" for val in row))
# %%
