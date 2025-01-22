# #%%
# import os
# import pathlib

# import cv2
# import numpy as np

# image_path = 'Train/Train/Urban/masks_png/1395.png'
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# pixel_values = np.array(img, dtype=int)

# for row in pixel_values:
#     print(" ".join(f"{val:3}" for val in row))
# # %%

import os
from PIL import Image
import numpy as np

label_mapping = {
    6: 1,  # Forest -> 1
    4: 2,  # Water -> 2
    5: 3,  # Barren land -> 3
    2: 4,  # Human activity (Building) -> 4
    3: 4,  # Human activity (Road) -> 4
    7: 4,  # Human activity (Agriculture) -> 4
    1: 5,  # Background -> 5
    0: 0   # No-data -> 0 (Ignored)
}

colour_mapping = {
    1: (0, 255, 0),     # Forest - Bright Green
    2: (0, 0, 255),     # Water - Bright Blue
    3: (255, 165, 0),   # Barren land - Bright Orange
    4: (255, 0, 255),   # Human activity - Bright Magenta
    5: (165, 165, 165),   # Background - Gray
    0: (0, 0, 0)        # No-data - Black
}

grayscale_mapping = {
    1: 50,   # Forest - Light gray
    2: 100,  # Water - Medium gray
    3: 150,  # Barren land - Darker gray
    4: 200,  # Human activity - Even darker gray
    5: 255,  # Background - White
    0: 0     # No-data - Black
}

# input_dir = 'Train/Train/Urban/masks_png'
# output_dir = 'TrainMasksColour'
# output_dir = 'TrainMasksGrayscale'

input_dir = 'Val/Val/Urban/masks_png'
output_dir = 'ValMasksColour'
# output_dir = 'ValMasksGrayscale'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        mask = Image.open(input_path).convert('L')
        mask_array = np.array(mask)

        processed_mask = np.zeros_like(mask_array, dtype=np.uint8)
        for original_label, new_label in label_mapping.items():
            processed_mask[mask_array == original_label] = new_label

        # grayscale to colour, uncomment when generating colour images
        colour_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
        for label, color in colour_mapping.items():
            colour_mask[processed_mask == label] = color

        colour_image = Image.fromarray(colour_mask, mode='RGB')
        colour_image.save(output_path)

        # # gray scale
        # grayscale_mask = np.zeros_like(processed_mask, dtype=np.uint8)
        # for label, gray_value in grayscale_mapping.items():
        #     grayscale_mask[processed_mask == label] = gray_value

        # colour_image = Image.fromarray(grayscale_mask, mode='L')
        # colour_image.save(output_path)

        print(f"Processed: {filename} -> {output_path}")

print("Processing complete, all masks saved to:", output_dir)
