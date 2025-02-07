import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from pathlib import Path

output_root = 'demo'
output_dir = Path(output_root)
if not output_dir.exists:
    output_dir.mkdir()

grayscale_mapping = {
    1: 50,  # Forest - Even darker gray
    2: 100,  # Water - Darker gray
    3: 150,  # Barren land - Medium gray
    4: 200,  # Human activity - Light gray
    5: 225,  # Agriculture - Even Lighter gray
    6: 255,  # Background - White
    0: 0  # No-data - Black
}

# Load the change detection model
change_detection_model_path = "C:/Users/User/Desktop/Python/deep_learning/Deep_Learning-Based_Change_Detection_of_Urban_Landscape/models/50_dsifn_fitted.pth"
change_detection_model = torch.jit.load(change_detection_model_path)
change_detection_model.eval()

# Load the semantic segmentation model
# segmentation_model_path = 'C:/Users/User/Desktop/Python/deep_learning/Deep_Learning-Based_Change_Detection_of_Urban_Landscape/models/13_deeplabv3_fitted.pth'
# segmentation_model_path = "D:/storage/deeplabv3/20_deeplabv3_fitted.pth"
# segmentation_model_path = "D:/storage/dpl/5_deeplabv3_fitted.pth"
segmentation_model_path = "D:/storage/dplbv3/16_deeplabv3_fitted.pth"
segmentation_model = torch.jit.load(segmentation_model_path)
segmentation_model.eval()

# Define preprocessing for change detection
change_detection_preprocess = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Define preprocessing for semantic segmentation
segmentation_preprocess = transforms.Compose([
    transforms.Resize((1024, 1024),
                      interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load pre-change and post-change images for change detection
# post_change_image = Image.open(
#     'D:/storage/ChangeDetectionDataset/Real/subset/val/A/00029.jpg').convert(
#         'RGB')
# pre_change_image = Image.open(
#     'D:/storage/ChangeDetectionDataset/Real/subset/val/B/00029.jpg').convert(
#         'RGB')
post_change_image = Image.open(
    'D:/storage/ChangeDetectionDataset/Real/subset/test/A/01508.jpg').convert(
        'RGB')
pre_change_image = Image.open(
    'D:/storage/ChangeDetectionDataset/Real/subset/test/B/01508.jpg').convert(
        'RGB')
# post_change_image = Image.open("C:/Users/User/Downloads/2_A.png").convert('RGB')
# pre_change_image = Image.open("C:/Users/User/Downloads/2_B.png").convert('RGB')
# post_change_image = Image.open("C:/Users/User/Downloads/2_Ab.png").convert(
#     'RGB')
# pre_change_image = Image.open("C:/Users/User/Downloads/2_Bb.png").convert(
#     'RGB')

# Preprocess images for change detection
pre_change_image_processed = change_detection_preprocess(
    post_change_image).unsqueeze(0)
post_change_image_processed = change_detection_preprocess(
    pre_change_image).unsqueeze(0)

# Move images and model to device
device = torch.device("cpu")
pre_change_image_processed = pre_change_image_processed.to(device)
post_change_image_processed = post_change_image_processed.to(device)
change_detection_model = change_detection_model.to(device)

# Perform change detection
with torch.no_grad():
    outputs = change_detection_model(pre_change_image_processed,
                                     post_change_image_processed)
    change_map = outputs[4].squeeze().cpu().numpy()

# Save the predicted change map
change_map_image = Image.fromarray((change_map * 255).astype(np.uint8))
change_map_image.save('demo/predicted_change_map.png')


def map_grayscale(value):
    return grayscale_mapping.get(value, 255)


vectorized_map = np.vectorize(map_grayscale)


def semantic_segmentation(segmentation_model, image, prefix='pre'):
    # Perform semantic segmentation
    with torch.no_grad():
        output = segmentation_model(image)
        predicted_mask = torch.argmax(output['out'].squeeze(),
                                      dim=0).byte().cpu().numpy()

    # Save the segmentation result
    segmentation_result = vectorized_map(predicted_mask)
    image = Image.fromarray(segmentation_result)
    image.save(f'demo/{prefix}_segmentation_result.png')

    return segmentation_result


# Preprocess the satellite image for semantic segmentation
satellite_image_processed = segmentation_preprocess(
    pre_change_image).unsqueeze(0)
satellite_image_processed_post = segmentation_preprocess(
    post_change_image).unsqueeze(0)

# Move image and model to device
satellite_image_processed_pre = satellite_image_processed.to(device)
satellite_image_processed_post = satellite_image_processed_post.to(device)
segmentation_model = segmentation_model.to(device)

pre_change_segmentation = semantic_segmentation(segmentation_model,
                                                satellite_image_processed_pre)
post_change_segmentation = semantic_segmentation(
    segmentation_model, satellite_image_processed_post, prefix='post')

predict_mask = np.array(Image.open('demo/post_segmentation_result.png'))

# Resize the change map to match the segmentation result dimensions
change_map_resized = np.array(
    Image.fromarray(change_map).resize((1024, 1024), Image.NEAREST))

# Overlay the change map on the segmentation result
overlay_image_post = np.where(change_map_resized > 0.5, predict_mask, 0)

# Save the overlay image
overlay_image_pil_post = Image.fromarray(overlay_image_post.astype(np.uint8))

overlay_image_pil_post.save('demo/overlay_image.png')

# Calculate the percentage of change
changed_pixels = np.sum(change_map_resized > 0.5)
total_pixels = change_map_resized.size
percentage_change = (changed_pixels / total_pixels) * 100

print(f"Percentage of change: {percentage_change:.2f}%")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Create a custom colormap for the legend
cmap = plt.cm.gray
norm = plt.Normalize(vmin=0, vmax=255)

# Create legend patches
legend_patches = [
    mpatches.Patch(color=cmap(norm(value)), label=f'{key}: {label}')
    for key, (label, value) in enumerate([('No-data', 0), (
        'Forest', 50), ('Water', 100), ('Barren land',
                                        150), ('Human activity',
                                               200), ('Agriculture',
                                                      225), ('Background',
                                                             255)])
]

# Visualize the results
plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.title('Pre-change Image')
plt.imshow(pre_change_image)

plt.subplot(3, 2, 2)
plt.title('Post-change Image')
plt.imshow(post_change_image)

plt.subplot(3, 2, 3)
plt.title('Pre-change Segmentation')
plt.imshow(pre_change_segmentation, cmap='gray', vmin=0, vmax=255)

plt.subplot(3, 2, 4)
plt.title('Post-change Segmentation')
plt.imshow(post_change_segmentation, cmap='gray', vmin=0, vmax=255)

plt.subplot(3, 2, 5)
plt.title('Predicted Change Map')
plt.imshow(change_map_image, cmap='gray', vmin=0, vmax=255)

plt.subplot(3, 2, 6)
plt.title('Overlay Image')
plt.imshow(overlay_image_pil_post, cmap='gray', vmin=0, vmax=255)

# Add the legend
plt.figlegend(handles=legend_patches, )

plt.tight_layout()
plt.show()
