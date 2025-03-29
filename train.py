"""Training noise2image model on experimental paired data of noise events and intensity images."""
import os
from argparse import ArgumentParser
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as l
from lightning.pytorch import loggers
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torchmetrics

from models.unet_attention import Unet
from models.resunet import ResUnet
import np_transforms
import utils

INDIST_EVENT_PATH = "/content/drive/MyDrive/Noise2iImageData/indist_events/"
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
parser.add_argument("--checkpoint_path", type=str, default='', 
                    help="Path to the checkpoint to load from. Will skip training if provided.")
parser.add_argument("--time_bin", type=int, default=1, 
                    help="Time binning during the event aggregation. Note that default value is 1, which means all events are aggregated into a single time bin.")
parser.add_argument("--pixel_bin", type=int, default=2, 
                    help="Pixel binning during the event aggregation.")
parser.add_argument("--polarity", action='store_true', 
                    help="Aggregate events into 2 channels for positive and negative polarities.")
parser.add_argument("--time_std", action='store_true', 
                    help="Add a channel for standard deviation of the timestamp.")
parser.add_argument("--integration_time_s", type=float, default=1, 
                    help="Event aggregation time in seconds.")

torch.set_float32_matmul_precision('medium')


class Model(l.LightningModule):
    def __init__(self, dim, in_channels, lr, vanilla_unet=False, ):
        super().__init__()

        if vanilla_unet:
            self.model = ResUnet(in_channels=in_channels, out_channels=1, dim=dim, conv_kernel_size=3)
        else:
            self.model = Unet(
                dim=dim,
                dim_mults=(1, 2, 4, 8),
                in_channels=in_channels,
                out_channels=1,
                flash_attn=True,
            )

        self.lr = lr
        self.running_sum = 0

        self.valid_metrics = torchmetrics.MetricCollection({
            'valid_psnr': torchmetrics.image.PeakSignalNoiseRatio(),
            'valid_ssim': torchmetrics.image.StructuralSimilarityIndexMeasure()
        })
        self.test_metrics = torchmetrics.MetricCollection({
            'test_psnr': torchmetrics.image.PeakSignalNoiseRatio(),
            'test_ssim': torchmetrics.image.StructuralSimilarityIndexMeasure()
        })
        self.save_hyperparameters()

    def forward(self, x, time):
        return self.model(x, time=time)

    def on_train_epoch_start(self):
        self.running_sum = 0
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        self.running_sum = 0
        return super().on_validation_epoch_start()

    def training_step(self, batch, batch_idx):
        x, y, t = batch
        y_hat = self.model(x, t)
        loss = (y_hat - y).pow(2).mean()
        self.running_sum += loss.item()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_train_loss', self.running_sum / (batch_idx + 1), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, t = batch
        y_hat = self.model(x, time=t)
        loss = (y_hat - y).pow(2).mean()
        self.running_sum += loss.item()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.valid_metrics(y_hat, y)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            self.logger.experiment.add_images('noise', np.tile((torch.sum(x, dim=1, keepdim=True).cpu()), (1, 3, 1, 1)),
                                              self.current_epoch)
            self.logger.experiment.add_images('reconstruction', np.tile((y_hat.cpu()), (1, 3, 1, 1)),
                                              self.current_epoch)
            self.logger.experiment.add_images('truth', np.tile((y.cpu()), (1, 3, 1, 1)), self.current_epoch)
        self.log('avg_val_loss', self.running_sum / (batch_idx + 1), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, t = batch
        y_hat = self.model(x, time=t)
        loss = (y_hat - y).pow(2).mean()
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)

        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)

        return loss

    def predict_step(self, batch):
        if len(batch) == 2:
            x, t = batch
        else:
            x, y, t = batch
        return self.model(x, t)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


if __name__ == '__main__':
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    input_size = (720 // args.pixel_bin // 8 * 8, 1280 // args.pixel_bin // 8 * 8)
    val_ds = utils.EventImagePairDataset(image_folder=INDIST_IMAGE_PATH,
                                         event_folder=INDIST_EVENT_PATH,
                                         integration_time_s=args.integration_time_s if args.integration_time_s > 0 else 1,
                                         total_time_s=10, start_time_s=5, time_bin=args.time_bin,
                                         pixel_bin=args.pixel_bin,
                                         polarity=args.polarity,
                                         std_channel=args.time_std,
                                         transform=transforms.Compose([np_transforms.CenterCrop(input_size),
                                                                       utils.EventCountNormalization()]))
    _, val_ds, test_ds = utils.data_split(val_ds, validation_split=0.1, testing_split=0.15, seed=47)

    tb_logger = loggers.tensorboard.TensorBoardLogger('lightning_logs', name=args.log_name)
    trainer = Trainer(
        logger=tb_logger,
        callbacks=[ModelCheckpoint(monitor='val_loss', save_top_k=2, save_last=True, mode='min', every_n_epochs=1),
                   LearningRateMonitor(logging_interval='epoch')],
        accelerator='gpu',
        devices=[args.gpu_ind, ],
        max_epochs=args.num_epochs,
    )

    if args.checkpoint_path != '':
        model = Model.load_from_checkpoint(args.checkpoint_path)
        print("loaded from checkpoint: ", args.checkpoint_path)
    else:
        in_channels = args.time_bin * 2 if args.polarity else args.time_bin
        if args.time_std:
            in_channels += 1
        model = Model(dim=64, in_channels=in_channels, lr=args.lr, vanilla_unet=args.vanilla_unet)

        ds = utils.EventImagePairDataset(image_folder=INDIST_IMAGE_PATH,
                                         event_folder=INDIST_EVENT_PATH,
                                         integration_time_s=args.integration_time_s, total_time_s=10, start_time_s=-1,
                                         time_bin=args.time_bin,
                                         pixel_bin=args.pixel_bin,
                                         polarity=args.polarity,
                                         std_channel=args.time_std,
                                         transform=transforms.Compose([
                                             np_transforms.RandomHorizontalFlip(),
                                             np_transforms.RandomVerticalFlip(),
                                             np_transforms.CenterCrop(input_size),
                                             utils.EventCountNormalization()
                                         ]))
        train_ds, _, _ = utils.data_split(ds, validation_split=0.1, testing_split=0.15, seed=47)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True)
        trainer.fit(model, train_dl,
                    DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                               persistent_workers=True))
        print("training finished")

    print("In-distribution testing:")
    trainer.test(model, DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                   persistent_workers=True))

    predictions = trainer.predict(model, dataloaders=DataLoader(test_ds, batch_size=args.batch_size,
                                                                num_workers=args.num_workers, shuffle=False,
                                                                persistent_workers=True))

    ood_ds = utils.EventImagePairDataset(image_folder=OOD_IMAGE_PATH,
                                         event_folder=OOD_EVENT_PATH,
                                         integration_time_s=args.integration_time_s if args.integration_time_s > 0 else 1,
                                         total_time_s=10, start_time_s=5, time_bin=args.time_bin,
                                         pixel_bin=args.pixel_bin,
                                         polarity=args.polarity,
                                         std_channel=args.time_std,
                                         transform=transforms.Compose([np_transforms.CenterCrop(input_size),
                                                                       utils.EventCountNormalization()]),
                                         img_suffix='.png',
                                         calib_img_path=os.path.join(OOD_EVENT_PATH, 'checkerboard0.png'))

    print("Out-of-distribution testing:")
    trainer.test(model, DataLoader(ood_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                   persistent_workers=True))
    predictions_ood = trainer.predict(model, dataloaders=DataLoader(ood_ds, batch_size=args.batch_size,
                                                                    num_workers=args.num_workers, shuffle=False,
                                                                    persistent_workers=True))
    np.savez(os.path.join(tb_logger.log_dir, 'predictions.npz'), pred=np.concatenate(predictions),
             pred_ood=np.concatenate(predictions_ood))
    print("predictions saved to ", os.path.join(tb_logger.log_dir, 'predictions.npz'))
