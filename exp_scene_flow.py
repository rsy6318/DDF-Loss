import os
import numpy as np
import torch

import open3d as o3d

import pytorch3d.ops
from model import setup_seed
from loss import CLGD,evaluate_3d

import argparse
from tqdm import tqdm


def optimize_flow(src:np.array,tgt:np.array,beta:float=20.0,epochs=1000,loss_func=None):
    src=torch.from_numpy(src).unsqueeze(0).cuda().float()
    tgt=torch.from_numpy(tgt).unsqueeze(0).cuda().float()
    flow=torch.zeros_like(src).float()

    flow.requires_grad=True

    optimizer=torch.optim.Adam([flow],lr=5e-2)
    for _ in tqdm(range(epochs)):
        pred_tgt=src+flow
        knn_index=pytorch3d.ops.knn_points(src,src,K=30,return_nn=True,return_sorted=True)[1]
        knn_flow=pytorch3d.ops.knn_gather(flow,knn_index)
        loss=loss_func(pred_tgt,tgt)+beta*torch.mean(torch.square(flow.unsqueeze(2)-knn_flow[:,:,1:,:])) #*1e-1 #+torch.mean(torch.exp(-dists_tgt[:,1:]/0.05))*1e-3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return flow.squeeze(0).cpu().detach().numpy()

if __name__=='__main__':
    setup_seed(1)
    pc1=np.loadtxt('example/scene_flow/pc1.xyz')
    pc2=np.loadtxt('example/scene_flow/pc2.xyz')
    gt_flow=np.loadtxt('example/scene_flow/gt_flow.xyz')

    loss_func=CLGD(weighted_query=False)
    pred_flow=optimize_flow(pc1,pc2,50,1000,loss_func)
    epe,acc_005,acc_01,outlier=evaluate_3d(pred_flow,gt_flow)
    print('EPE: %0.4f, ACC Relax (0.05): %0.4f, ACC Strict (0.1): %0.4f, Outlier: %0.4f'%(epe,acc_005,acc_01,outlier))
