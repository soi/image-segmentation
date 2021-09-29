#! /usr/bin/env python3
import os
import math
import time
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from torchmetrics import Precision, Recall
from torchsummary import summary


class BreastCancerDataset(Dataset):
    def __init__(self, args, split, transforms):
        assert split in ['train', 'valid', 'test']
        self.args = args
        self.split = split
        self.transforms = transforms

        start_time = time.time()
        print(f'\nloading {self.split} data into memory ...')

        ids = [l.strip() for l in open(os.path.join(self.args.data_dir, f'ids_{split}'))]
        self.X = self._load_images(ids, 'Images', '_ccd.tif')
        self.y = self._load_images(ids, 'Masks', '.TIF')
        self.y[self.y == 255] = 1  # set labels to 0 and 1

        print(f'Shape X: {self.X.shape}')
        print(f'Shape y: {self.y.shape}')

    def _load_images(self, ids, folder, suffix):
        images = [np.array(Image.open(os.path.join(self.args.data_dir, folder, f'{id_}{suffix}')))
                  for id_ in ids]
        images = np.stack(images, axis=0)
        if folder == 'Images':
            # move channels to the front for the inputs
            images = images.swapaxes(3, 2).swapaxes(2, 1)
        return images

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transforms(torch.Tensor(self.X[idx])), self.y[idx]


class BreastCancerDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.transform_normalize = transforms.Normalize([203.09668496, 177.48439689, 221.1715344],
                                                        [54.13497255, 63.98501953, 30.16765668])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # set data augmentation
            train_aug_list = [self.transform_normalize]
            if self.args.aug_crop:
                train_aug_list.append(transforms.RandomCrop(size=(self.args.crop_len, self.args.crop_len)))
            if self.args.aug_rotate:
                train_aug_list.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
            if self.args.aug_colorjitter:
                train_aug_list.append(transforms.ColorJitter(brightness=random.random(), contrast=random.random()))
            self.transforms_train = transforms.Compose(train_aug_list)
            self.transforms_valid = transforms.Compose([self.transform_normalize])

    def train_dataloader(self):
        train_dataset = BreastCancerDataset(self.args, 'train', self.transforms_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                      num_workers=self.args.num_workers, drop_last=True)
        self.train_dataloader = train_dataloader
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = BreastCancerDataset(self.args, 'valid', self.transforms_valid)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False,
                                    num_workers=self.args.num_workers, drop_last=False)
        self.valid_dataloader = valid_dataloader
        return valid_dataloader


class BreastCancerModel(pl.LightningModule):
    def __init__(self, args):

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        super().__init__()
        self.args = args
        self.preds = []

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = torch.sigmoid(x)  # output value between 0 and 1 for BCE
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def calc_loss(self, y_pred, y_true, bce_weight=0.5, threshold=0.5, prefix='train'):
        def dice_loss(y_pred, y_true, smooth=1.):
            iflat = y_pred.view(-1)
            tflat = y_true.view(-1)
            intersection = (iflat * tflat).sum()
            return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

        y_pred = y_pred.squeeze()
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        # threshold y_pred after bce loss calculation
        y_pred = (y_pred > threshold).cast(torch.int8)

        dice_loss = dice_loss(y_pred, y_true)
        total_loss = bce_loss * bce_weight + dice_loss * (1 - bce_weight)

        self.log(f'{prefix}_loss_total', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{prefix}_loss_bce', bce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{prefix}_loss_dice', dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        y_pred = self.forward(x)
        loss = self.calc_loss(y_pred, y_true, prefix='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        _ = self.calc_loss(y_pred, y_true, prefix='train')

        y_pred = y_pred.detach()
        if y_pred.is_cuda:
            y_pred = y_pred.cpu()
        self.preds.append(np.argmax(y_pred.numpy(), axis=1))


class BreastCancerCallbacks(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        dataset = trainer.datamodule.valid_dataloader.dataset
        print(f'\n\ntrain_loss: {metrics["train_loss"]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, default='../data/real_data/')
    # training params
    parser.add_argument('-bs', '--batch-size', type=int, default=8)
    parser.add_argument('-e', '--epochs', type=int, default=500)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--aug-rotate', action='store_true')
    parser.add_argument('--aug-colorjitter', action='store_true')
    parser.add_argument('--aug-crop', action='store_true')
    parser.add_argument('--crop-len', type=int, default=512)
    # general setup
    parser.add_argument('-nw', '--num-workers', type=int, default=0)
    parser.add_argument('-p', '--patience', type=int, default=20)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print()

    data_module = BreastCancerDataModule(args)
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[BreastCancerCallbacks(args)],
                                            max_epochs=args.epochs,
                                            num_sanity_val_steps=0,
                                            log_every_n_steps=1)
    model = BreastCancerModel(args)
    model.to('cuda')

    summary(model, input_size=(3, 512, 512))

    trainer.fit(model, data_module)
