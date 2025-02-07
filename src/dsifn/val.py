from pathlib import Path
from datahandler import get_val_dataloader_single_folder

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from sklearn.metrics import f1_score, jaccard_score

from tqdm import tqdm


def mIOU(prediction, label, num_classes):
    # prediction = prediction.max(1)[1].float().cpu().numpy()
    # label = label.float().cpu().numpy()

    iou_list = list()
    present_iou_list = list()

    for sem_class in range(num_classes):
        pred_inds = (prediction == sem_class)
        target_inds = (label == sem_class)
        if target_inds.sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).sum().item()
            union_now = pred_inds.sum().item() + target_inds.sum().item(
            ) - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    miou = np.mean(present_iou_list)
    return miou


def deep_supervision_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


def val_model(model, dataloaders, metrics):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    model.eval()

    running_loss = 0.0

    phase = 'Val'

    f1_record = []
    mIoU_record = []

    # Iterate over data.
    for sample in tqdm(iter(dataloaders[phase])):
        pre_change_image = sample['pre_image'].to(device)
        post_change_image = sample['post_image'].to(device)
        masks = sample['mask'].to(device)

        # track history if only in train
        with torch.set_grad_enabled(False):
            side_outputs = model(pre_change_image, post_change_image)

            transform1 = transforms.Compose([
                transforms.Resize((15, 15)),
            ])
            transform2 = transforms.Compose([
                transforms.Resize((30, 30)),
            ])
            transform3 = transforms.Compose([
                transforms.Resize((61, 61)),
            ])
            transform4 = transforms.Compose([
                transforms.Resize((122, 122)),
            ])

            target1 = transform1(masks)
            target2 = transform2(masks)
            target3 = transform3(masks)
            target4 = transform4(masks)

            loss1 = deep_supervision_loss(side_outputs[0], target1)
            loss2 = deep_supervision_loss(side_outputs[1], target2)
            loss3 = deep_supervision_loss(side_outputs[2], target3)
            loss4 = deep_supervision_loss(side_outputs[3], target4)
            loss5 = deep_supervision_loss(side_outputs[4], masks)

            total_loss = loss1 + loss2 + loss3 + loss4 + loss5

            y_pred = side_outputs[4].data.cpu().numpy().ravel()
            y_true = masks.data.cpu().numpy().ravel()
            for name, metric in metrics.items():
                if name == 'f1_score':
                    f1 = metric(y_true > 0, y_pred > 0.1)
                    f1_record.append(f1)
                elif name == 'mIoU':
                    miou = metric(side_outputs[4].data.cpu().numpy(), masks.data.cpu().numpy(), 6)
                    mIoU_record.append(miou)
                # else:
                #     if len(np.unique(y_true)
                #            ) > 1:  # Check if both classes are present
                #         batchsummary[f'{phase}_{name}'].append(
                #             metric(y_true.astype('uint8'), y_pred))
                #     else:
                #         batchsummary[f'{phase}_{name}'].append(0)

            running_loss += total_loss.item()

    epoch_loss = running_loss / len(dataloaders[phase])

    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    print(f'F1 Score: {np.mean(f1_record):,.5f}')
    print(f'mIoU Score: {np.mean(mIoU_record):,.5f}')
    print(
        f'Loss1: {loss1:,.2f}, Loss2: {loss2:,.2f}, Loss3: {loss3:,.2f}, Loss4: {loss4:,.2f}, Loss5: {loss5:,.2f}'
    )


def main():
    data_directory = Path('D:/storage/ChangeDetectionDataset/Real/subset/val')
    exp_directory = Path(
        'C:/Users/User/Desktop/Python/deep_learning/Deep_Learning-Based_Change_Detection_of_Urban_Landscape/src/dsifn/output/val'
    )
    if not exp_directory.exists():
        exp_directory.mkdir()

    train_dataloader = get_val_dataloader_single_folder(
        data_dir=data_directory,
        pre_image_folder='A',
        post_image_folder='B',
        mask_folder='OUT',
    )

    model = torch.jit.load(
        'C:/Users/User/Desktop/Python/deep_learning/Deep_Learning-Based_Change_Detection_of_Urban_Landscape/models/50_dsifn_fitted.pth'
    )
    model.eval()

    metrics = {'f1_score': f1_score}

    _ = val_model(model=model, dataloaders=train_dataloader, metrics=metrics)


if __name__ == "__main__":
    main()
