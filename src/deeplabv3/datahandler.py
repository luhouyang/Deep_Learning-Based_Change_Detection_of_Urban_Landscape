"""
Link: https://github.com/msminhas93/DeepLabv3FineTuning/blob/master/datahandler.py
"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from segdataset import SegmentationDataset


def get_dataloader_single_folder(
        data_dir: str,
        image_folder: str = 'Images',
        mask_folder: str = 'Masks',
        #  fraction: float = 0.175,
        fraction: float = 0.04,
        batch_size: int = 2):
    """Create train and test dataloader from a single directory containing
    the image and mask folders.

    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'. 
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Test set. Defaults to 0.175.
        batch_size (int, optional): Dataloader batch size. Defaults to 2.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])  # values determined according to pytorch documentation

    image_datasets = {
        x:
        SegmentationDataset(data_dir,
                            image_folder=image_folder,
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
