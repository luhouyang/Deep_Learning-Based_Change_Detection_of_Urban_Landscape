"""
Link: https://github.com/msminhas93/DeepLabv3FineTuning/blob/master/datahandler.py
"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from segdataset import SegmentationDataset


def get_dataloader_sep_folder(data_dir: str,
                              pre_image_folder: str = 'B',
                              post_image_folder: str = 'A',
                              mask_folder: str = 'OUT',
                              batch_size: int = 4):
    """ Create Train and Test dataloaders from two
        separate Train and Test folders.
        The directory structure should be as follows.
        data_dir
        --Train
        ------Image
        ---------Image1
        ---------ImageN
        ------Mask
        ---------Mask1
        ---------MaskN
        --Test
        ------Image
        ---------Image1
        ---------ImageM
        ------Mask
        ---------Mask1
        ---------MaskM

    Args:
        data_dir (str): The data directory or root.
        image_folder (str, optional): Image folder name. Defaults to 'Image'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Mask'.
        batch_size (int, optional): Batch size of the dataloader. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_datasets = {
        x:
        SegmentationDataset(root=Path(data_dir) / x,
                            transforms=data_transforms,
                            pre_image_folder=pre_image_folder,
                            post_image_folder=post_image_folder,
                            mask_folder=mask_folder)
        for x in ['Train', 'Test']
    }
    dataloaders = {
        x:
        DataLoader(image_datasets[x],
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=8)
        for x in ['Train', 'Test']
    }
    return dataloaders


def get_dataloader_single_folder(data_dir: str,
                                 pre_image_folder: str = 'B',
                                 post_image_folder: str = 'A',
                                 mask_folder: str = 'Masks',
                                 fraction: float = 0.175,
                                 batch_size: int = 4):
    """Create train and test dataloader from a single directory containing
    the image and mask folders.

    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'. 
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Test set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """

    data_transforms = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_datasets = {
        x:
        SegmentationDataset(data_dir,
                            pre_image_folder=pre_image_folder,
                            post_image_folder=post_image_folder,
                            mask_folder=mask_folder,
                            seed=100,
                            fraction=fraction,
                            subset=x,
                            transforms=data_transforms)
        for x in ['Train', 'Test']
    }
    dataloaders = {
        x:
        DataLoader(image_datasets[x],
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=16)
        for x in ['Train', 'Test']
    }
    return dataloaders


def get_val_dataloader_single_folder(data_dir: str,
                                     pre_image_folder: str = 'B',
                                     post_image_folder: str = 'A',
                                     mask_folder: str = 'Masks',
                                     batch_size: int = 1):

    data_transforms = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_datasets = {
        'Val':
        SegmentationDataset(data_dir,
                            pre_image_folder='B',
                            post_image_folder='A',
                            mask_folder=mask_folder,
                            seed=100,
                            fraction=0.0,
                            subset='Val',
                            transforms=data_transforms)
    }
    dataloaders = {
        'Val':
        DataLoader(image_datasets['Val'],
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=16)
    }
    return dataloaders
