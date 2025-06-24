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
from models.resunet import SNNResUnet, DEFAULT_BETA as MODEL_DEFAULT_BETA
import np_transforms
import utils as utils

INDIST_EVENT_PATH = "/content/drive/MyDrive/Noise2ImageData/indist_events/"
INDIST_IMAGE_PATH = "/content/drive/MyDrive/Noise2ImageData/indist_images/"
OOD_EVENT_PATH = "/content/drive/MyDrive/Noise2ImageData/ood_DIV2K_events/"
OOD_IMAGE_PATH = "/content/drive/MyDrive/Noise2ImageData/ood_DIV2K_images/"

parser = ArgumentParser()
parser.add_argument("--gpu_ind", type=int, default=0)
parser.add_argument("--vanilla_unet", action='store_true',
                    help="If true, uses SNNResUnet")
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=1) 
parser.add_argument("--num_workers", type=int, default=2) 
parser.add_argument("--log_name", type=str, default='')
parser.add_argument("--checkpoint_path", type=str, default='')
parser.add_argument("--time_bin", type=int, default=1)
parser.add_argument("--pixel_bin", type=int, default=2)
parser.add_argument("--polarity", action='store_true')
parser.add_argument("--time_std", action='store_true')
parser.add_argument("--integration_time_s", type=float, default=1)
parser.add_argument("--snn_num_steps", type=int, default=40)
parser.add_argument("--model_dim", type=int, default=32)
parser.add_argument("--snn_conv_kernel_size", type=int, default=3)
parser.add_argument("--snn_beta", type=float, default=MODEL_DEFAULT_BETA)

torch.set_float32_matmul_precision('medium')

class Model(l.LightningModule):
    def __init__(self, model_dim, in_channels, lr, vanilla_unet=False, 
                 snn_num_steps=40, snn_conv_kernel_size=3, snn_beta=MODEL_DEFAULT_BETA):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.vanilla_unet:
            self.model = SNNResUnet(
                in_channels=self.hparams.in_channels,
                out_channels=1,
                dim=self.hparams.model_dim,
                conv_kernel_size=self.hparams.snn_conv_kernel_size,
                beta=self.hparams.snn_beta
            )
        else:
            self.model = Unet(
                dim=self.hparams.model_dim,
                dim_mults=(1, 2, 4, 8),
                in_channels=self.hparams.in_channels,
                out_channels=1,
                flash_attn=True,
            )

        self.running_sum = 0
        self.valid_metrics = torchmetrics.MetricCollection({
            'valid_psnr': torchmetrics.image.PeakSignalNoiseRatio(),
            'valid_ssim': torchmetrics.image.StructuralSimilarityIndexMeasure()
        })
        self.test_metrics = torchmetrics.MetricCollection({
            'test_psnr': torchmetrics.image.PeakSignalNoiseRatio(),
            'test_ssim': torchmetrics.image.StructuralSimilarityIndexMeasure()
        })

    def forward(self, x, time):
        return self.model(x, time=time)

    def on_train_epoch_start(self):
        self.running_sum = 0
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        self.running_sum = 0
        return super().on_validation_epoch_start()

    def _get_time_argument_for_model(self, t_if_ann):
        if isinstance(self.model, SNNResUnet):
            return self.hparams.snn_num_steps
        return t_if_ann

    def training_step(self, batch, batch_idx):
        x, y, t_from_batch = batch
        time_arg = self._get_time_argument_for_model(t_from_batch)
        y_hat = self(x, time=time_arg) 

        loss = (y_hat - y).pow(2).mean() 
        self.running_sum += loss.item()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_train_loss', self.running_sum / (batch_idx + 1), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, t_from_batch = batch
        time_arg = self._get_time_argument_for_model(t_from_batch)
        y_hat = self(x, time=time_arg) 
        
        loss = (y_hat - y).pow(2).mean()
        self.running_sum += loss.item()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.valid_metrics(y_hat, y)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            log_reconstruction = y_hat.detach().cpu()
            if log_reconstruction.ndim == 4 and log_reconstruction.shape[1] == 1:
                 log_reconstruction = np.tile(log_reconstruction, (1, 3, 1, 1)) #check tmrw

            self.logger.experiment.add_images('noise', np.tile((torch.sum(x, dim=1, keepdim=True).cpu()), (1, 3, 1, 1)),
                                              self.current_epoch)
            if log_reconstruction.ndim == 4:
                self.logger.experiment.add_images('reconstruction', log_reconstruction, self.current_epoch)
            self.logger.experiment.add_images('truth', np.tile((y.cpu()), (1, 3, 1, 1)), self.current_epoch)
        self.log('avg_val_loss', self.running_sum / (batch_idx + 1), on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, t_from_batch = batch
        time_arg = self._get_time_argument_for_model(t_from_batch)
        y_hat = self(x, time=time_arg) 
        loss = (y_hat - y).pow(2).mean()
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)

        
        self.test_metrics(y_hat, y) 
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx=None):
        t_from_batch_default = self.hparams.snn_num_steps if isinstance(self.model, SNNResUnet) else 1
        
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3: 
                x, _, t_from_batch_val = batch
            elif len(batch) == 2: 
                x, t_from_batch_val = batch
            elif len(batch) == 1: 
                x = batch[0]
                t_from_batch_val = t_from_batch_default
            else: 
                x = batch 
                t_from_batch_val = t_from_batch_default
        else: 
            x = batch 
            t_from_batch_val = t_from_batch_default
        
        time_arg = self._get_time_argument_for_model(t_from_batch_val)
        return self(x, time=time_arg) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.99))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

if __name__ == '__main__':
    args = parser.parse_args()

    input_size = (720 // args.pixel_bin // 8 * 8, 1280 // args.pixel_bin // 8 * 8)

    val_ds = utils.EventImagePairDataset(image_folder=INDIST_IMAGE_PATH,
                                         event_folder=INDIST_EVENT_PATH,
                                         integration_time_s=args.integration_time_s if args.integration_time_s > 0 else 1,
                                         total_time_s=10, start_time_s=5, time_bin=args.time_bin,
                                         pixel_bin=args.pixel_bin, polarity=args.polarity,
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
        precision='16-mixed'
    )

    if args.checkpoint_path != '':
        in_channels_calc = args.time_bin * 2 if args.polarity else args.time_bin
        if args.time_std:
            in_channels_calc += 1
        
        model = Model.load_from_checkpoint(
            args.checkpoint_path,
            map_location='cpu', 
            model_dim=args.model_dim,
            in_channels=in_channels_calc, 
            lr=args.lr,
            vanilla_unet=args.vanilla_unet, 
            snn_num_steps=args.snn_num_steps,
            snn_conv_kernel_size=args.snn_conv_kernel_size,
            snn_beta=args.snn_beta 
        )
        if torch.cuda.is_available() and hasattr(args, 'gpu_ind'):
             model = model.to(f'cuda:{args.gpu_ind}')
        print("loaded from checkpoint: ", args.checkpoint_path)
    else:
        in_channels = args.time_bin * 2 if args.polarity else args.time_bin
        if args.time_std:
            in_channels += 1
        model = Model(model_dim=args.model_dim, 
                      in_channels=in_channels,
                      lr=args.lr,
                      vanilla_unet=args.vanilla_unet,
                      snn_num_steps=args.snn_num_steps,
                      snn_conv_kernel_size=args.snn_conv_kernel_size,
                      snn_beta=args.snn_beta) 

        train_event_image_path_ds = utils.EventImagePairDataset(image_folder=INDIST_IMAGE_PATH,
                                         event_folder=INDIST_EVENT_PATH,
                                         integration_time_s=args.integration_time_s, total_time_s=10, start_time_s=-1,
                                         time_bin=args.time_bin, pixel_bin=args.pixel_bin,
                                         polarity=args.polarity, std_channel=args.time_std,
                                         transform=transforms.Compose([
                                             np_transforms.RandomHorizontalFlip(),
                                             np_transforms.RandomVerticalFlip(),
                                             np_transforms.CenterCrop(input_size),
                                             utils.EventCountNormalization()]))
        train_ds, _, _ = utils.data_split(train_event_image_path_ds, validation_split=0.1, testing_split=0.15, seed=47)
        
        use_persistent_workers = args.num_workers > 0
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, 
                              persistent_workers=use_persistent_workers, shuffle=True, 
                              pin_memory=torch.cuda.is_available())
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, 
                            persistent_workers=use_persistent_workers, 
                            pin_memory=torch.cuda.is_available())
        
        trainer.fit(model, train_dl, val_dl)
        print("training finished")

    use_persistent_workers_test = args.num_workers > 0
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, 
                         persistent_workers=use_persistent_workers_test, 
                         pin_memory=torch.cuda.is_available())
    print("In-distribution testing:")
    trainer.test(model, dataloaders=test_dl)

    predictions_list = trainer.predict(model, dataloaders=test_dl)
    if predictions_list:
        predictions = torch.cat([p.cpu() for p in predictions_list], dim=0).numpy()
    else:
        predictions = np.array([])

    ood_ds = utils.EventImagePairDataset(image_folder=OOD_IMAGE_PATH,
                                         event_folder=OOD_EVENT_PATH,
                                         integration_time_s=args.integration_time_s if args.integration_time_s > 0 else 1,
                                         total_time_s=10, start_time_s=5, time_bin=args.time_bin,
                                         pixel_bin=args.pixel_bin, polarity=args.polarity,
                                         std_channel=args.time_std,
                                         transform=transforms.Compose([np_transforms.CenterCrop(input_size),
                                                                       utils.EventCountNormalization()]),
                                         img_suffix='.png',
                                         calib_img_path=os.path.join(OOD_EVENT_PATH, 'checkerboard0.png'))
    
    ood_dl = DataLoader(ood_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, 
                        persistent_workers=use_persistent_workers_test, 
                        pin_memory=torch.cuda.is_available())
    print("Out-of-distribution testing:")
    trainer.test(model, dataloaders=ood_dl)
    predictions_ood_list = trainer.predict(model, dataloaders=ood_dl)

    if predictions_ood_list:
        predictions_ood = torch.cat([p.cpu() for p in predictions_ood_list], dim=0).numpy()
    else:
        predictions_ood = np.array([])
            
    if tb_logger.log_dir and predictions.size > 0 :
        save_path = os.path.join(tb_logger.log_dir, 'predictions.npz')
        if predictions_ood.size > 0:
            np.savez(save_path, pred=predictions, pred_ood=predictions_ood)
        else:
            np.savez(save_path, pred=predictions)
        print("predictions saved to ", save_path)
    else:
        print("No predictions to save or logger path not available.")
