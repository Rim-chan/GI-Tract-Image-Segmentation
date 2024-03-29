import torch
import torch.nn as nn
import numpy as np
import gc
from losses import LossUWGITract
from metrics import UWGITractMetrics
import pytorch_lightning as pl
from monai.networks.nets import DynUNet

class Unet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.build_model()
        self.loss = LossUWGITract()
        self.metrics = UWGITractMetrics(n_class=self.args.out_channels, 
                                        shape = self.args.resize_shape) 
    

    def training_step(self, batch, batch_idx):
        img, lbl = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, lbl = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)
        self.metrics.update(logits, lbl, loss)
        
    def predict_step(self, batch, batch_idx):
        img, lbl = batch
        preds = self.model(img)
        preds = (nn.Sigmoid()(preds) > 0.5).int()
        lbl_np = lbl.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        np.save(self.args.save_path + 'predictions.npy', preds_np)
        np.save(self.args.save_path + 'labels.npy', lbl_np)

        
    def training_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        gc.collect()
        
    def validation_epoch_end(self, outputs):
        dice, hausdorff, loss, eval_metric = self.metrics.compute()
        dice_mean = dice.mean().item()
        hausdorff_mean = hausdorff.mean().item()
        eval_metric_mean = eval_metric.mean().item()
        self.metrics.reset()
        print(f"Val_Performace: Mean_Dice {dice_mean:.3f}, Mean_Hausdorff {hausdorff_mean:.3f}, \
                Val_Loss {loss.item():.3f}, Evaluation Metric {eval_metric_mean:.3f}")
        self.log("dice_mean", dice_mean)
        self.log("hausdorff_mean", hausdorff_mean)
        self.log("eval_metric_mean", eval_metric_mean)
        torch.cuda.empty_cache()
        gc.collect()

    def build_model(self):
        self.model = DynUNet(
            spatial_dims=2,
            in_channels=self.args.in_channels,
            out_channels=self.args.out_channels,
            kernel_size=self.args.kernels,
            strides=self.args.strides,
            upsample_kernel_size=self.args.strides[1:],
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01})
        )
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)