import glob # mandatory for the GF
import math
import numpy as np
import pandas as pd
import skimage.io as io
from torchvision.transforms import CenterCrop
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import albumentations as A
import cv2


class UWGITractDataset(Dataset): 
    def __init__(self, base_dir, csv_path, crop_size):
        self.scans = sorted(glob.glob(base_dir + '/**/*.png', recursive=True))
        self.train_csv = pd.read_csv(csv_path)
        self.crop_size = crop_size
        self.no_nans = self.drop_nans(self.train_csv)
        self.files_with_nones = [self.retrieve_masks(idx) for idx in range(len(self.scans))]
        self.files = [i for i in self.files_with_nones if i]

    def drop_nans(self, df):
        df['segmentation'] = df.segmentation.fillna('')
        df['length'] = df.segmentation.map(len)
        df2 = df.groupby('id')['segmentation'].agg(list).to_frame().reset_index()
        df2 = df2.merge(df.groupby('id')['length'].agg(sum).to_frame().reset_index())
        df2.drop(df2[df2.length == 0].index, inplace=True) #16590 
        df2['missing'] = df2.segmentation.map(lambda x : True if '' in x else False)
        df2.drop(df2[df2.missing == True].index, inplace=True)
        return df2  
    
    def retrieve_masks(self, idx):
        bits = self.scans[idx].split('/')
        mask_id = '_'.join([bits[-3]] + bits[-1].split('_')[:2])
        
        if mask_id in self.no_nans['id'].to_list():
            masks_row = self.no_nans.loc[self.no_nans['id'] == mask_id]['segmentation'].item()
            large_bowel_str, small_bowel_str, stomach_str = masks_row
            return  {'scan':self.scans[idx],
                             'stomach' : stomach_str,
                             'small_bowel': small_bowel_str,
                             'large_bowel': large_bowel_str}

    def rl_decoder(self, shape, rl):
        s = rl.split()
        msk = np.zeros((shape[0]*shape[1]), dtype='uint8')
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for (lo, hi) in zip(starts, ends):
            msk[lo:hi] = 1
        return msk.reshape(shape)
    
    def __len__(self): 
        return len(self.files) 

    def __getitem__(self, idx):
        file = self.files[idx]
        scan = io.imread(file['scan']).astype('float32')
        
        mx = scan.max()
        if mx:
            scan = scan / mx
        
        mask_stomach = self.rl_decoder(scan.shape, file['stomach'])
        mask_small_bowel = self.rl_decoder(scan.shape, file['small_bowel'])
        mask_large_bowel = self.rl_decoder(scan.shape, file['large_bowel'])
        
        lbl = np.stack([mask_stomach, mask_small_bowel, mask_large_bowel], axis=2)
        transforms = A.Compose([A.Resize(224, 224, interpolation=cv2.INTER_NEAREST)], p=1.0)
        data = transforms(image=scan[..., None], mask=lbl)
        img = torch.tensor(data['image'].transpose((2, 0, 1)), dtype=torch.float32)
        lbl = torch.tensor(data['mask'].transpose((2, 0, 1)), dtype=torch.float32)
        return img, lbl

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
        return DataLoader(self.uwgitract_train, batch_size=self.args.batch_size, 
                          num_workers=self.args.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.uwgitract_val, batch_size=self.args.batch_size, 
                          num_workers=self.args.num_workers, drop_last=False)