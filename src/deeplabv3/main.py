import time
import torch
import copy
import csv
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from datahandler import get_dataloader_single_folder
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from sklearn.metrics import f1_score, jaccard_score, accuracy_score


def createDeepLabv3():
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True,
        progress=True,
        weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier = DeepLabHead(2048, num_classes=6)

    model.train()
    return model


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    # check for gpu
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

            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device).squeeze(1).long()

                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    # loss = criterion(outputs['out'], masks.squeeze(1).long())
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy()
                    y_pred = y_pred.argmax(1).ravel()
                    y_true = masks.data.cpu().numpy().ravel()

                    for name, metric in metrics.items():
                        if name == 'accuracy_score':
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))

        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])

        print(batchsummary)

        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test':
                if loss < best_loss:
                    best_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                model_path = f'{epoch}_deeplabv3_fitted.pth'
                model_scripted = torch.jit.script(model)
                model_scripted.save(model_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    data_directory = Path('D:/storage/loveda/Train/Train/Urban')
    exp_directory = Path(
        'C:/Users/User/Desktop/Python/deep_learning/Deep_Learning-Based_Change_Detection_of_Urban_Landscape/src/deeplabv3/output'
    )
    if not exp_directory.exists():
        exp_directory.mkdir()

    train_dataloader = get_dataloader_single_folder(
        data_dir=data_directory,
        image_folder='images_png',
        mask_folder='TrainMasksGrayscale')

    model = createDeepLabv3()
    model.train()

    # criterion = torch.nn.MSELoss(reduction='mean')
    criterion = torch.nn.CrossEntropyLoss()
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    metrics = {'accuracy_score': accuracy_score}

    epochs = 20

    history = train_model(model=model,
                          criterion=criterion,
                          dataloaders=train_dataloader,
                          optimizer=optimizer,
                          bpath=exp_directory,
                          metrics=metrics,
                          num_epochs=epochs)


if __name__ == '__main__':
    main()
