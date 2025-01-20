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
import cv2
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

# open-cv (BGR) formatting
color_mapping = {
    1: (0, 255, 0),     # Forest - Bright Green
    2: (255, 0, 0),     # Water - Bright Blue
    3: (0, 165, 255),   # Barren land - Bright Orange
    4: (255, 0, 255),   # Human activity - Bright Magenta
    5: (165, 165, 165), # Background - Gray
    0: (0, 0, 0)        # No-data - Black
}

# input_dir = 'Train/Train/Urban/masks_png'
# output_dir = 'TrainMasksColour'
# output_dir = 'TrainMasksGrayscale'

input_dir = 'Val/Val/Urban/masks_png'
# output_dir = 'ValMasksColour'
output_dir = 'ValMasksGrayscale'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # map labels
        processed_mask = np.zeros_like(mask, dtype=np.uint8)
        for original_label, new_label in label_mapping.items():
            processed_mask[mask == original_label] = new_label

        # # grayscale to colour, uncomment when generating colour images
        # colour_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # for label, color in color_mapping.items():
        #     colour_mask[processed_mask == label] = color

        # save
        # cv2.imwrite(output_path, colour_mask)
        cv2.imwrite(output_path, processed_mask)

        print(f"Processed: {filename} -> {output_path}")

print("Processing complete, all masks saved to:", output_dir)

