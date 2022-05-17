import torch
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
        self.metrics = UWGITractMetrics(n_class=self.args.out_channels) 
        
    def forward(self, img):
        return torch.argmax(self.model(img, dim=1))
    
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
        
    def training_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        gc.collect()
        
    def validation_epoch_end(self, outputs):
        dice, hausdorff, loss = self.metrics.compute()
        dice_mean = dice.mean().item()
        hausdorff_mean = hausdorff.mean().item()
        self.metrics.reset()
        print(f"Val_Performace: Mean_Dice {dice_mean}, Mean_Hausdorff {hausdorff_mean}, Val_Loss {loss.item()}")
        self.log("dice_mean", dice_mean)
        self.log("hausdorff_mean", hausdorff_mean)
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