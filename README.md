# **Deep Learning-Based Change Detection of Forest, Water, Barren Land, and Human Activity Growth**

Keywords: deep learning; transfer learning; forest cover change detection; river change detection; barren land detection; very high resolution (VHR); DeepLabv3+; deeply supervised image fusion network (DSIFN); desertification

## **Dataset**

[Notes about data preparation](/data_preparation)

1. [LoveDA](https://github.com/Junjue-Wang/LoveDA) | [Processed](https://drive.google.com/drive/folders/1AX5DdNeSseyn3rN89jYoNEznxX7QCUgH?usp=drive_link)

1. [Change Detection Dataset](https://isprs-archives.copernicus.org/articles/XLII-2/565/2018/)

## **Run training loop**

1. `cd` to the directory with main.py file
    ```
    cd ../src/dsifn
    ```

1. Pass [data directory](https://isprs-archives.copernicus.org/articles/XLII-2/565/2018/), output directory, number of epochs as arguments in commandline
    ```
    python main.py data_dir output_dir epochs
    ```

## **Report Draft**

[Report](https://drive.google.com/file/d/1YIqk1mqUxTfQ6spSek5eOrf2dWRm4xFW/view?usp=sharing)

## **Models**

Deeply Supervised Image Fusion Network [PyTorch .pth Model](https://drive.google.com/file/d/1FvhzXGa9grV2fcWcrTcKyfRg9HVwf81y/view?usp=sharing) | [CODE](/src/dsifn/)

To run Deeply Supervised Image Fusion Network model

```
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

model = torch.jit.load('model.pth')
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                          0.225])
])

pre_change_image = Image.open('Pre-Change_Image.jpg')
post_change_image = Image.open('Post-Change_Image.jpg').convert('RGB')

pre_change_image = preprocess(pre_change_image).unsqueeze(0)
post_change_image = preprocess(post_change_image).unsqueeze(0)

device = torch.device("cpu")
pre_change_image = pre_change_image.to(device)
post_change_image = post_change_image.to(device)
model = model.to(device)

with torch.no_grad():
    outputs = model(pre_change_image, post_change_image)
    change_map = outputs[4].squeeze().cpu().numpy()

ground_truth_mask = Image.open('Ground_Truth_Change_Mask.jpg').convert('RGB').convert('L')

ground_truth_mask = ground_truth_mask.resize((244, 244))
ground_truth_mask = transforms.ToTensor()(ground_truth_mask).squeeze().numpy()

change_map_image = Image.fromarray((change_map * 255).astype(np.uint8))
change_map_image.save('test/predicted_change_map.png')

# display
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.imshow(pre_change_image.squeeze().permute(1, 2, 0).cpu().numpy())
plt.title('Pre-change Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(post_change_image.squeeze().permute(1, 2, 0).cpu().numpy())
plt.title('Post-change Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(ground_truth_mask, cmap='gray')
plt.title('Ground Truth Mask')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(change_map, cmap='gray')
plt.axis('off')

plt.show()
```

Fine-Tuned Deeplabv3 [PyTorch .pth Model](https://drive.google.com/file/d/12cCxIMtiiuOSRVBpbgVe5j1sS2l0lRsy/view?usp=sharing) | [CODE](/src/deeplabv3/)

To run Fine-Tuned Deeplabv3 model

```
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

model_path = 'model.pth'
model = torch.jit.load(model_path)
model.eval()

device = torch.device("cpu")
model.to(device)

image_path = 'VHR_Satellite_Image.png'
true_mask = 'Ground_Truth_Segmentation_Mask.png'
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize to match model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = transform(image).unsqueeze(0)
inputs = image.to(device)

with torch.no_grad():
    output = model(inputs)

predicted_mask = torch.argmax(output['out'].squeeze(),
                              dim=0).byte().cpu().numpy()

output_image = Image.fromarray(predicted_mask)
output_image.save('segmentation_result.png')

mask_true = Image.open(true_mask).convert('L')
ground_truth_mask = mask_true.resize((1024, 1024))
ground_truth_mask = transforms.ToTensor()(ground_truth_mask).squeeze().numpy()

# display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(Image.open(image_path))
axes[0].set_title('Original Image')

axes[1].imshow(ground_truth_mask, cmap='gray')
axes[1].set_title('Ground Truth Mask')

axes[2].imshow(predicted_mask, cmap='gray')
axes[2].set_title(f'Predicted Mask')

plt.show()
```

## **DSIFN**

![DSIFN prediction image](/src/dsifn/output/Figure_1.png)

## **Setup/Environment**

[Notes about setup & environment](/docs/setup)
