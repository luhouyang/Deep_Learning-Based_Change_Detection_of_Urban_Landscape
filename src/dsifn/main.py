import copy
import csv
import os
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from datahandler import get_dataloader_single_folder

import torch.nn.functional as F


def deep_supervision_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction), nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool, _ = torch.max(x, dim=2, keepdim=True)
        max_pool, _ = torch.max(max_pool, dim=3, keepdim=True)

        avg_pool = self.mlp(avg_pool.view(x.size(0),
                                          -1)).view(x.size(0), -1, 1, 1)
        max_pool = self.mlp(max_pool.view(x.size(0),
                                          -1)).view(x.size(0), -1, 1, 1)
        attention = self.sigmoid(avg_pool + max_pool)
        return x * attention


class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concatenated = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(concatenated))
        return x * attention


class DSIFN(torch.nn.Module):

    def __init__(self):
        super(DSIFN, self).__init__()

        # feature Extraction CNN Model, VGG16 up to MaxPool5
        self.vgg = models.vgg16(weights='DEFAULT')
        self.vgg_feature_layers = [4, 9, 16, 23, 30]

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.vgg_model_fe = torch.nn.Sequential(
            *list(self.vgg.features.children())[:30])

        # print("Feature Extraction CNN Model, VGG16 up to MaxPool5")
        # torchinfo.summary(self.vgg_model_fe, (3, 244, 244), batch_dim=0)

        # DSIFN model layers
        self.channel_attentions = nn.ModuleList([
            ChannelAttention(1536),  # feature spaces = f23 * 2 + f30
            ChannelAttention(768),  # feature spaces = f16 * 2 + f23
            ChannelAttention(384),  # feature spaces = f9 * 2 + f16
            ChannelAttention(192)  # feature spaces = f4 * 2 + f9
        ])

        self.spatial_attentions = nn.ModuleList([
            SpatialAttention(),
            SpatialAttention(),
            SpatialAttention(),
            SpatialAttention(),
            SpatialAttention()
        ])

        self.sigmoid = nn.Sigmoid()

        # deep supervision
        self.ds_conv2d = nn.ModuleList([
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Conv2d(32, 1, kernel_size=1)
        ])

        self.upsampling = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),  # 15 -> 30
            nn.Upsample(size=(61, 61), mode='bilinear',
                        align_corners=True),  # 30 -> 61
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),  # 61 -> 122
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),  # 122 -> 244
        ])

        self.conv1 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, 3, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, 3, padding='same'),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(1536, 1024, 3, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(1024, 512, 3, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 256, 3, padding='same'),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(768, 512, 3, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 256, 3, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 128, 3, padding='same'),
                                   nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(384, 256, 3, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 128, 3, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 64, 3, padding='same'),
                                   nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(192, 128, 3, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 64, 3, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 32, 3, padding='same'),
                                   nn.ReLU())

    def extract_features(self, x):
        # returns [f4, f9, f16, f23, f30]
        features = []
        for i, layer in enumerate(self.vgg_model_fe):
            x = layer(x)
            if i + 1 in self.vgg_feature_layers:
                features.append(x)
        return features

    def forward(self, x1, x2):
        """
        x1: pre-change image
        x2: post-change image

        pass through feature extractor first
        [f4, f9, f16, f23, f30] = [64, 128, 256, 512, 512]
        """
        # stream 1 pre-change
        x1_f4, x1_f9, x1_f16, x1_f23, x1_f30 = self.extract_features(x1)

        # stream 2 post-change
        x2_f4, x2_f9, x2_f16, x2_f23, x2_f30, = self.extract_features(x2)

        #
        """
        Concat - f30_x1 + f30_x2 = [1, 1024, 15, 15]
        Conv2D - (1024, 512, 3)
        Conv2D - (512, 512, 3)
        Conv2D - (512, 512, 3)
        SAM_1 - SpatialAttention()
        DS_1 - Conv2D(512, 1, 1) -> Sigmoid()
        """
        concat_1 = torch.cat((x1_f30, x2_f30), dim=1)
        idf_1 = self.conv1(concat_1)
        idf_1 = self.spatial_attentions[0](idf_1)

        ds_1 = self.ds_conv2d[0](idf_1)
        ds_1 = self.sigmoid(ds_1)

        #
        """
        Up_IDF_1 - Upsample(2)
        Concat - Up_IDF_1 + x1_f30 + x2_f23 = [1, 1536, 30, 30]
        CAM_1 - ChannelAttention(1536)
        Conv2D - (1536, 1024, 3)
        Conv2D - (1024, 512, 3)
        Conv2D - (512, 256, 3)
        SAM_2 - SpatialAttention()
        DS_2 - Conv2D(256, 1, 1) -> Sigmoid()
        """
        up_idf_1 = self.upsampling[0](idf_1)
        concat_2 = torch.cat((up_idf_1, x1_f23, x2_f23), dim=1)
        idf_2 = self.channel_attentions[0](concat_2)
        idf_2 = self.conv2(idf_2)
        idf_2 = self.spatial_attentions[1](idf_2)

        ds_2 = self.ds_conv2d[1](idf_2)
        ds_2 = self.sigmoid(ds_2)

        #
        """
        Up_IDF_2 - Upsample((61, 61))
        Concat - Up_IDF_3 + x1_f16_x1 + x2_f16 = [1, 768, 61, 61]
        CAM_2 - ChannelAttention(768)
        Conv2D - (768, 512, 3)
        Conv2D - (512, 256, 3)
        Conv2D - (256, 128, 3)
        SAM_3 - SpatialAttention()
        DS_3 - Conv2D(128, 1, 1) -> Sigmoid()
        """
        up_idf_2 = self.upsampling[1](idf_2)
        concat_3 = torch.cat((up_idf_2, x1_f16, x2_f16), dim=1)
        idf_3 = self.channel_attentions[1](concat_3)
        idf_3 = self.conv3(idf_3)
        idf_3 = self.spatial_attentions[2](idf_3)

        ds_3 = self.ds_conv2d[2](idf_3)
        ds_3 = self.sigmoid(ds_3)

        #
        """
        Up_IDF_3 - Upsample(2)
        Concat - Up_IDF_3 + x1_f9 + x2_f9 = [1, 384, 122, 122]
        CAM_3 - ChannelAttention(384)
        Conv2D - (384, 256, 3)
        Conv2D - (256, 128, 3)
        Conv2D - (128, 64, 3)
        SAM_4 - SpatialAttention()
        DS_4 - Conv2D(64, 1, 1) -> Sigmoid()
        """
        up_idf_3 = self.upsampling[2](idf_3)
        concat_4 = torch.cat((up_idf_3, x1_f9, x2_f9), dim=1)
        idf_4 = self.channel_attentions[2](concat_4)
        idf_4 = self.conv4(idf_4)
        idf_4 = self.spatial_attentions[3](idf_4)

        ds_4 = self.ds_conv2d[3](idf_4)
        ds_4 = self.sigmoid(ds_4)

        #
        """
        Up_IDF_4 - Upsample(2)
        Concat - Up_IDF_4 + x1_f4 + x2_f4 = [1, 192, 244, 244]
        CAM_4 - ChannelAttention(192)
        Conv2D - (192, 128, 3)
        Conv2D - (128, 64, 3)
        Conv2D - (64, 32, 3)
        SAM_5 - SpatialAttention()
        Change Map - Conv2D(32, 1, 1) -> Sigmoid()
        """
        up_idf_4 = self.upsampling[3](idf_4)
        concat_5 = torch.cat((up_idf_4, x1_f4, x2_f4), dim=1)
        idf_5 = self.channel_attentions[3](concat_5)
        idf_5 = self.conv5(idf_5)
        idf_5 = self.spatial_attentions[4](idf_5)

        ds_5 = self.ds_conv2d[4](idf_5)
        ds_5 = self.sigmoid(ds_5)
        """
        return deep supervision outputs, for back-propogation from intermediate layers
        """
        return (ds_1, ds_2, ds_3, ds_4, ds_5)


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs, scheduler):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                pre_change_image = sample['pre_image'].to(device)
                post_change_image = sample['post_image'].to(device)
                masks = sample['mask'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
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

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        # total_loss.backward()
                        loss1.backward(retain_graph=True)
                        loss2.backward(retain_graph=True)
                        loss3.backward(retain_graph=True)
                        loss4.backward(retain_graph=True)
                        loss5.backward()

                        optimizer.step()

                    y_pred = side_outputs[4].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        # else:
                        #     if len(np.unique(y_true)
                        #            ) > 1:  # Check if both classes are present
                        #         batchsummary[f'{phase}_{name}'].append(
                        #             metric(y_true.astype('uint8'), y_pred))
                        #     else:
                        #         batchsummary[f'{phase}_{name}'].append(0)

                    running_loss += total_loss.item()

            batchsummary['epoch'] = epoch
            epoch_loss = running_loss / len(dataloaders[phase])
            batchsummary[f'{phase}_loss'] = epoch_loss
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print(
                f'Loss1: {loss1:,.2f}, Loss2: {loss2:,.2f}, Loss3: {loss3:,.2f}, Loss4: {loss4:,.2f}, Loss5: {loss5:,.2f}'
            )

        scheduler.step(epoch_loss)

        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)

        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                model_path = f'{epoch}_dsifn_fitted.pth'
                model_scripted = torch.jit.script(model)
                model_scripted.save(model_path)

            if epoch % 5 == 0:
                model_path = f'{epoch}_dsifn_fitted.pth'
                model_scripted = torch.jit.script(model)
                model_scripted.save(model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model_path = 'finished_dsifn_fitted.pth'
    model_scripted = torch.jit.script(model)
    model_scripted.save(model_path)
    return model


def main():
    data_directory = Path(
        'D:/storage/ChangeDetectionDataset/Real/subset/train')
    exp_directory = Path(
        'C:/Users/User/Desktop/Python/deep_learning/Deep_Learning-Based_Change_Detection_of_Urban_Landscape/src/dsifn/output'
    )
    if not exp_directory.exists():
        exp_directory.mkdir()

    train_dataloader = get_dataloader_single_folder(data_dir=data_directory,
                                                    pre_image_folder='A',
                                                    post_image_folder='B',
                                                    mask_folder='OUT')

    model = DSIFN()
    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.9,
                                                     patience=5,
                                                     verbose=True)

    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    epochs = 50

    _ = train_model(model=model,
                    criterion=criterion,
                    dataloaders=train_dataloader,
                    optimizer=optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs,
                    scheduler=scheduler)


if __name__ == "__main__":
    main()
