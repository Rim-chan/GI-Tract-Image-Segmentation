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
        self.no_nans = self.drop_nans(self.train_csv)

    def drop_nans(self, df):
        df['segmentation'] = df.segmentation.fillna('')
        df['length'] = df.segmentation.map(len)
        df2 = df.groupby('id')['segmentation'].agg(list).to_frame().reset_index()
        df2 = df2.merge(df.groupby('id')['length'].agg(sum).to_frame().reset_index())
        df2.drop(df2[df2.length ==0].index, inplace=True) #16590 
        return df2  
    
    def retrieve_masks(self, idx):
        bits = self.scans[idx].split('\\')
        mask_id = '_'.join([bits[-3]] + bits[-1].split('_')[:2])
        
        masks_row = self.no_nans.loc[self.no_nans['id'] == mask_id]['segmentation'].item()
        large_bowel_str, small_bowel_str, stomach_str = masks_row
        
        return large_bowel_str, small_bowel_str, stomach_str 
    
    def combine_files(self, idx):
        large_bowel_str, small_bowel_str, stomach_str = self.retrieve_masks(idx)
        files = {'scan':self.scans[idx],
                 'stomach' : stomach_str,
                 'small_bowel': small_bowel_str,
                 'large_bowel': large_bowel_str}
        return files    
    
    def rl_decoder(self, shape, rl):
        H, W = shape
        mask = np.zeros((H * W))
        
        if rl == '':
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
        scan = scan/ np.iinfo(scan.dtype).max
        mask_stomach = self.rl_decoder(scan.shape, file['stomach'])
        mask_small_bowel = self.rl_decoder(scan.shape, file['small_bowel'])
        mask_large_bowel = self.rl_decoder(scan.shape, file['large_bowel'])
        img = torch.tensor(scan, dtype=torch.float32)
        lbl = torch.tensor(np.stack([mask_stomach, mask_small_bowel, mask_large_bowel]), dtype=torch.uint8)
        transforms = CenterCrop(size=self.crop_size)
        return transforms(img)[None], transforms(lbl)





class UWGITractDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
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