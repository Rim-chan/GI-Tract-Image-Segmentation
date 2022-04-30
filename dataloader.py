import glob # mandatory for the GF
import math
import numpy as np
import pandas as pd
import skimage.io as io
from torchvision.transforms import CenterCrop
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import Trainer, LightningDataModule

class UWGITractDataset(Dataset):
    def __init__(self, base_dir, csv_path, crop_size):
        self.scans = sorted(glob.glob(base_dir + '\\**\\*.png', recursive=True))
        self.train_csv = pd.read_csv(csv_path)
        self.crop_size = crop_size
        
    def retrieve_masks(self, idx):
        bits = self.scans[idx].split('\\')
        mask_id = '_'.join([bits[-3]] + bits[-1].split('_')[:2])
        
        masks_df = self.train_csv.loc[self.train_csv['id'] == mask_id]
        stomach_row = masks_df.loc[masks_df['class'] == 'stomach']
        small_bowel_row = masks_df.loc[masks_df['class'] == 'small_bowel']
        large_bowel_row = masks_df.loc[masks_df['class'] == 'large_bowel']
        
        return stomach_row, small_bowel_row, large_bowel_row
    
    def combine_files(self, idx):
        stomach_row, small_bowel_row, large_bowel_row = self.retrieve_masks(idx)
        files = {'scan':self.scans[idx],
                 'stomach' : stomach_row,
                 'small_bowel': small_bowel_row,
                 'large_bowel': large_bowel_row}
        return files    
    
    def rl_decoder(self, shape, row):
        H, W = shape
        mask = np.zeros((H * W))
        rl = row['segmentation'].values.item()
        
        if not isinstance(rl, str):
            return mask.reshape(H,W)
        else:
            rl = rl.split(' ')
            for i in range(0, len(rl) - 1, 2):
                mask[int(rl[i]) - 1 : int(rl[i]) - 1 + int(rl[i + 1])] = 1
            return mask.reshape(H, W)
    
    def __len__(self):
        return len(self.scans)
    
    def __getitem__(self, idx):
        file = self.combine_files(idx)
        scan = io.imread(file['scan']).astype(np.int16)
        mask_stomach = self.rl_decoder(scan.shape, file['stomach'])
        mask_small_bowel = self.rl_decoder(scan.shape, file['small_bowel'])
        mask_large_bowel = self.rl_decoder(scan.shape, file['large_bowel'])
        img = torch.tensor(scan, dtype=torch.float32)
        lbl = torch.tensor(np.stack([mask_stomach, mask_small_bowel, mask_large_bowel]), dtype=torch.uint8)
        transforms = CenterCrop(size=self.crop_size)
        return transforms(img)[None], transforms(lbl)





class UWGITractDataModule(LightningDataModule):
    def __init__(self, args):
        self.args = args
    
    def setup(self, stage=None):
        uwgitract_dataset = UWGITractDataset(self.args.base_dir, self.args.csv_path, self.args.crop_size)
        self.uwgitract_train, self.uwgitract_val = random_split(uwgitract_dataset,
                                                            [math.ceil(0.85*len(uwgitract_dataset)),
                                                             math.floor(0.15*len(uwgitract_dataset))]) # ADD SEED LATER

    def train_dataloader(self):
        return DataLoader(self.uwgitract_train, batch_size=self.args.batch_size, num_workers=self.args.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.uwgitract_val, batch_size=self.args.batch_size, num_workers=self.args.num_workers, drop_last=False)