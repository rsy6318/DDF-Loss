import numpy as np
import torch
import torch.nn as nn
import itertools
import random
from LieAlgebra import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class MLP_Decoder(nn.Module):
    def __init__(self, feat_dims=512):
        super(MLP_Decoder, self).__init__()
        
        self.folding1 = nn.Sequential(
                nn.Conv1d(2, feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(feat_dims, feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(feat_dims, feat_dims, 1),
            )

        self.folding2 = nn.Sequential(
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, 3, 1),
        )




    def forward(self, points):
        # points (B,N,2)
        cat1 = points.transpose(1,2)   
        folding_result1 = self.folding1(cat1)           # (batch_size, 3, num_points)
        cat2 = folding_result1 
        folding_result2 = self.folding2(cat2)           # (batch_size, 3, num_points)
        return folding_result2.transpose(1, 2)          # (batch_size, num_points ,3)



def build_grid(res,delta=0.3):
    meshgrid = [[-delta, delta, res], [-delta, delta, res]]
    
    x = np.linspace(*meshgrid[0])
    y = np.linspace(*meshgrid[1])
    points = np.array(list(itertools.product(x, y))).astype(np.float32)    #(B,res*res,2)
    
    return points

class Reconstruction_point(nn.Module):
    def __init__(self, rotation=None, translation=None,zero_init=False):
        super(Reconstruction_point, self).__init__()
        if zero_init:
            tp = np.random.randn(3)
            tp = tp / np.linalg.norm(tp) * 0
            tp_translation = np.random.randn(3)* 0
            self.parameters_ = nn.Parameter(
                    torch.from_numpy(
                        np.concatenate([0.001 * tp, tp_translation],
                                    0).astype(np.float32)))
    
        else:
            if rotation is None or translation is None:
                tp = np.random.randn(3)
                tp = tp / np.linalg.norm(tp) #* 0
                tp_translation = np.random.randn(3)* 0.001
                self.parameters_ = nn.Parameter(
                    torch.from_numpy(
                        np.concatenate([0.001 * tp, tp_translation],
                                    0).astype(np.float32)))
            else:
                Trans = torch.zeros(4, 4)
                Trans[:3, :3] = rotation.reshape(3, 3)
                Trans[:3, 3] = translation.reshape(3)
                tp = torch.rand(6) * 0.6
                self.parameters_ = nn.Parameter(se3.log(Trans).reshape(-1) + tp)

    def Transform(self):
        return se3.exp3(self.parameters_)

    def forward(self, points, points_neighbors):
        R, T = self.Transform()
        update_points = points @ R + T.reshape(1, 1, 3)
        if points_neighbors is not None:
            points_neighbors = points_neighbors @ R + T.reshape(1, 1, 3)
            points_neighbors=points_neighbors.reshape(-1, 9)

        return update_points.reshape(-1, 3), points_neighbors