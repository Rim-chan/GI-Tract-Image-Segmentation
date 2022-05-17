import torch
import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss


class LossUWGITract(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
        self.ce = FocalLoss(gamma=2.0, to_onehot_y=False)
        
    def _loss(self, p, y):
        return self.dice(p, y) + self.ce(p, y)
    
    def forward(self, p, y):
        p_st, p_sb, p_lb = p[:, 0].unsqueeze(1), p[:, 1].unsqueeze(1), p[:, 2].unsqueeze(1)
        y_st, y_sb, y_lb = y[:, 0].unsqueeze(1), y[:, 1].unsqueeze(1), y[:, 2].unsqueeze(1)
        l_st, l_sb, l_lb = self._loss(p_st, y_st), self._loss(p_sb, y_sb), self._loss(p_lb, y_lb)
        return l_st + l_sb + l_lb