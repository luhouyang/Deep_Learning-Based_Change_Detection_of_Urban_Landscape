"""
Author: Manpreet Singh Minhas
Contact: msminhas at uwaterloo ca
"""
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms


class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """

    def __init__(self,
                 root: str,
                 pre_image_folder: str,
                 post_image_folder: str,
                 mask_folder: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale") -> None:
        """
        Args:
            root (str): Root directory path.
            pre_image_folder (str): Name of the folder that contains the pre-change images.
            post_image_folder (str): Name of folder that contains the post-change images.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.

        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)
        pre_image_folder_path = Path(self.root) / pre_image_folder
        post_image_folder_path = Path(self.root) / post_image_folder
        mask_folder_path = Path(self.root) / mask_folder
        if not pre_image_folder_path.exists():
            raise OSError(f"{pre_image_folder_path} does not exist.")
        if not post_image_folder_path.exists():
            raise OSError(f"{post_image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:
            self.pre_image_names = sorted(pre_image_folder_path.glob("*"))
            self.post_image_names = sorted(post_image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction
            self.pre_image_list = np.array(
                sorted(pre_image_folder_path.glob("*")))
            self.post_image_list = np.array(
                sorted(post_image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.pre_image_list))
                np.random.shuffle(indices)
                self.pre_image_list = self.pre_image_list[indices]
                self.post_image_list = self.post_image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == "Train":
                self.pre_image_names = self.pre_image_list[:int(
                    np.ceil(len(self.pre_image_list) * (1 - self.fraction)))]
                self.post_image_names = self.post_image_list[:int(
                    np.ceil(len(self.post_image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.pre_image_names = self.pre_image_list[int(
                    np.ceil(len(self.pre_image_list) * (1 - self.fraction))):]
                self.post_image_names = self.post_image_list[int(
                    np.ceil(len(self.post_image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[
                    int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self) -> int:
        return len(self.pre_image_names)

    def __getitem__(self, index: int) -> Any:
        pre_image_path = self.pre_image_names[index]
        post_image_path = self.post_image_names[index]
        mask_path = self.mask_names[index]
        with open(pre_image_path, "rb") as pre_image_file, open(
                post_image_path,
                "rb") as post_image_file, open(mask_path, "rb") as mask_file:

            pre_image = Image.open(pre_image_file)
            post_image = Image.open(post_image_file)

            if self.image_color_mode == "rgb":
                pre_image = pre_image.convert("RGB")
                post_image = post_image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                pre_image = pre_image.convert("L")
                post_image = post_image.convert("L")

            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")

            sample = {
                "pre_image": pre_image,
                "post_image": post_image,
                "mask": mask
            }

            if self.transforms:
                sample["pre_image"] = self.transforms(sample["pre_image"])
                sample["post_image"] = self.transforms(sample["post_image"])
                sample["mask"] = transforms.Compose([
                    transforms.Resize((244, 244)),
                    transforms.ToTensor(),
                ])(sample["mask"])

            return sample
