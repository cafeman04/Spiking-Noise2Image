"""Training noise2image model on synthetic data of noise events."""

from argparse import ArgumentParser
import os

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from lightning.pytorch import loggers
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from train import Model
import np_transforms
import utils

torch.set_float32_matmul_precision('medium')

INDIST_EVENT_PATH = "/content/drive/MyDrive/Noise2ImageData/indist_events/"
INDIST_IMAGE_PATH = "/content/drive/MyDrive/Noise2ImageData/indist_images/"
OOD_EVENT_PATH = "/content/drive/MyDrive/Noise2ImageData/ood_DIV2K_events/"
OOD_IMAGE_PATH = "/content/drive/MyDrive/Noise2ImageData/ood_DIV2K_images/"

parser = ArgumentParser()
parser.add_argument("--gpu_ind", type=int, default=0, help="GPU index")
parser.add_argument("--vanilla_unet", action='store_true', 
                    help="Use vanilla U-Net instead of the advanced u-net with attention layers")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=3, help="Batch size")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader")
parser.add_argument("--log_name", type=str, default='', 
                    help="Name of the log & checkpoint folder under ./lightning_logs.")
parser.add_argument("--pixel_bin", type=int, default=2, 
                    help="Pixel binning during the event aggregation.")
parser.add_argument("--time_bin", type=int, default=1, 
                    help="Time binning during the event aggregation. Note that default value is 1, which means all events are aggregated into a single time bin.")
parser.add_argument("--aug_contrast", action='store_true', 
                    help="Augment image contrast during synthetic training.")

num_photon_scalar= 1.0050
num_time= 21.9882
eps_pos = 0.9040
eps_neg = 1.0235
bias_pr= 1.7023
illum_offset = 0.1930
constant_noise_neg = 0.1398


if __name__ == '__main__':
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    input_size = (720 // args.pixel_bin // 8 * 8 * args.pixel_bin, 1280 // args.pixel_bin // 8 * 8 * args.pixel_bin)
    input_size_ds = [input_size[0]//args.pixel_bin, input_size[1]//args.pixel_bin]

    train_transform = [np_transforms.RandomHorizontalFlip(),
                       np_transforms.RandomVerticalFlip(),
                       np_transforms.CenterCrop(input_size),
                       utils.EventNoiseCountWrapper(num_photon_scalar=num_photon_scalar,
                                                    num_time=num_time, eps_pos=eps_pos,
                                                    eps_neg=eps_neg, bias_pr=bias_pr,
                                                    illum_offset=illum_offset,
                                                    constant_noise_neg=constant_noise_neg,
                                                    pixel_bin=args.pixel_bin, varying_eps=True),
                       utils.EventCountNormalization()]
    if args.aug_contrast:
        train_transform.insert(3, utils.AugmentImageContrast(max_scale=1.3, min_scale=0.7))

    ds = utils.EventImagePairDataset(image_folder=INDIST_IMAGE_PATH,
                                     event_folder=INDIST_EVENT_PATH,
                                     integration_time_s=1, total_time_s=10, start_time_s=5,
                                     time_bin=1, pixel_bin=1, polarity=True, std_channel=False,
                                     transform=transforms.Compose(train_transform))
    ds_train, _, ds_simu_test = utils.data_split(ds, validation_split=0.1, testing_split=0.15, seed=47)

    exp_ds = utils.EventImagePairDataset(image_folder=INDIST_IMAGE_PATH,
                                         event_folder=INDIST_EVENT_PATH,
                                         integration_time_s=1,
                                         total_time_s=10, start_time_s=5, time_bin=args.time_bin,
                                         pixel_bin=args.pixel_bin, polarity=True,
                                         transform=transforms.Compose([np_transforms.CenterCrop(input_size_ds),
                                                                       utils.EventCountNormalization()]))
    _, ds_val, test_ds = utils.data_split(exp_ds, validation_split=0.1, testing_split=0.15, seed=47)

    ood_ds = utils.EventImagePairDataset(image_folder=OOD_IMAGE_PATH,
                                         event_folder=OOD_EVENT_PATH,
                                         integration_time_s=1,
                                         total_time_s=10, start_time_s=5, time_bin=args.time_bin,
                                         pixel_bin=args.pixel_bin, polarity=True, std_channel=False,
                                         transform=transforms.Compose([np_transforms.CenterCrop(input_size_ds),
                                                                       utils.EventCountNormalization()]),
                                         img_suffix='.png',
                                         calib_img_path=os.path.join(OOD_EVENT_PATH, 'checkerboard0.png'))

    model = Model(dim=64, in_channels=args.time_bin * 2, lr=args.lr, vanilla_unet=args.vanilla_unet)
    tb_logger = loggers.tensorboard.TensorBoardLogger('lightning_logs', name=args.log_name)
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[ModelCheckpoint(monitor='val_loss', save_top_k=2, save_last=True, mode='min', every_n_epochs=1),
                   LearningRateMonitor(logging_interval='epoch')],
        accelerator='gpu',
        devices=[args.gpu_ind, ],
        max_epochs=args.num_epochs,
    )
    trainer.fit(model,
                DataLoader(ds_train, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True),
                DataLoader(ds_val, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, shuffle=False))

    trainer.test(model, DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                   persistent_workers=True))

    predictions = trainer.predict(model, dataloaders=DataLoader(test_ds, batch_size=args.batch_size,
                                                                num_workers=args.num_workers, shuffle=False,
                                                                persistent_workers=True))
    predictions_ood = trainer.predict(model, dataloaders=DataLoader(ood_ds, batch_size=args.batch_size,
                                                                    num_workers=args.num_workers, shuffle=False,
                                                                    persistent_workers=True))
    predictions_simu = trainer.predict(model, dataloaders=DataLoader(ds_simu_test, batch_size=args.batch_size,
                                                                     num_workers=args.num_workers, shuffle=False,
                                                                     persistent_workers=True))
    np.savez(os.path.join(tb_logger.log_dir, 'predictions.npz'),
             pred=np.concatenate(predictions),
             pred_simu=np.concatenate(predictions_simu),
             pred_ood=np.concatenate(predictions_ood))
    print("predictions saved to ", os.path.join(tb_logger.log_dir, 'predictions.npz'))
