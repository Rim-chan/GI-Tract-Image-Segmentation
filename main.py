from telnetlib import DONT
from dataloader import *
from args import *
from model import *
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer


if __name__ == "__main__":
    args = get_main_args()
    callbacks = []
    model = Unet(args)
    dm = UWGITractDataModule(args)
    model_ckpt = ModelCheckpoint(dirpath="./", filename="best_model",
                                monitor="dice_mean", mode="max", save_last=True)
    callbacks.append(model_ckpt)
    trainer = Trainer(callbacks=callbacks, enable_checkpointing=True, max_epochs=args.num_epochs, 
                enable_progress_bar=True, gpus=1, accelerator="gpu", amp_backend='apex', profiler='simple')

    # train the model
    if args.exec_mode == 'train':
        trainer.fit(model, dm)
    else:
        trainer.predict(model, datamodule=dm, ckpt_path=args.ckpt_path)  


        
    # dataset = UWGITractDataset(args.base_dir, args.csv_path, args.crop_size)
    # dm = UWGITractDataModule(args)
    # dm.setup()
    # img, lbl = next(iter(dm.train_dataloader()))
    # img, lbl = dataset[0]


    # print(img.shape, lbl.shape)


    # print(len(dm.train_dataloader()))