import torch
import math
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from monai.metrics.utils import get_mask_edges, get_surface_distance
from torchmetrics import Metric
                

class UWGITractMetrics(Metric):
    def __init__(self, n_class, shape):
        super().__init__(dist_sync_on_step=False)
        self.n_class = n_class
        self.max_dist = np.sqrt(shape ** 2 + shape ** 2)
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros(n_class), dist_reduce_fx="sum")
        self.add_state("hausdorff", default=torch.zeros(n_class), dist_reduce_fx="sum")
        self.add_state("loss", default=torch.zeros(1), dist_reduce_fx="sum")
        
    def update(self, p, y, loss):
        self.steps += 1
        self.dice += self.compute_dice(p, y)
        self.hausdorff += self.compute_hausdorff(p, y) 
        self.loss += loss
        
    def compute(self):
        mean_dice = self.dice / self.steps
        mean_hausdorff = self.hausdorff / self.steps
        mean_loss = self.loss / self.steps
        eval_metric = 0.4 * mean_dice + 0.6 * mean_hausdorff
        return 100 * mean_dice, mean_hausdorff, mean_loss, eval_metric
    
    def compute_dice(self, p, y):
        scores = torch.zeros(self.n_class, device=p.device, dtype=torch.float32)
        p = (torch.sigmoid(p) > 0.5).int()
        
        for i in range(self.n_class):
            p_i, y_i = p[:, i], y[:, i]
            if (y_i != 1).all():
                scores[i - 1] = 1 if (p_i != 1).all() else 0
                continue
            tp, fn, fp = self.get_stats(p_i, y_i, 1)
            denom = (2 * tp + fn + fp).to(torch.float)
            score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0
            scores[i - 1] = score_cls
        return scores

    def compute_hausdorff(self, p, y):
        scores = torch.zeros(self.n_class, device=p.device, dtype=torch.float32)
        p = (torch.sigmoid(p) > 0.5).int()
        for i in range(self.n_class):
            p_i, y_i = p[:, i], y[:, i]
            if torch.all(p_i == y_i):
                scores[i] = 0    
            (edges_p_i, edges_y_i) = get_mask_edges(p_i, y_i)
            surface_distance = get_surface_distance(edges_p_i, edges_y_i, distance_metric="euclidean")
            if surface_distance.shape == (0,):
                scores[i] = 0
            dist = surface_distance.max()
            if dist > self.max_dist:
                scores[i] = 1
            scores[i] =  dist / self.max_dist    
        return scores

    
    @staticmethod
    def get_stats(p, y, class_idx):
        tp = torch.logical_and(p == class_idx, y == class_idx).sum()
        fn = torch.logical_and(p != class_idx, y == class_idx).sum()
        fp = torch.logical_and(p == class_idx, y != class_idx).sum()
        return tp, fn, fp