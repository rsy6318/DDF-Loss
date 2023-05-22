import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
import open3d as o3d
from loss import CLGD,cal_error
from model import Reconstruction_point,setup_seed
from tqdm import tqdm

if __name__=='__main__':
    setup_seed(1)
    loss_func=CLGD()
    src=o3d.io.read_point_cloud('example/registration/src.ply')
    tgt=o3d.io.read_point_cloud('example/registration/tgt.ply')
    src=np.array(src.points)
    tgt=np.array(tgt.points)

    gt_trans=np.loadtxt('example/registration/gt_trans.txt')

    Reconstruction=Reconstruction_point(zero_init=True).cuda()

    optimizer=torch.optim.Adam(Reconstruction.parameters(),lr=2e-2)
    src=torch.from_numpy(src).cuda().float()
    tgt=torch.from_numpy(tgt).cuda().float()
    for epoch in tqdm(range(1,1000+1)):
        transformed_src,_=Reconstruction(src,None)
        
        loss=loss_func(transformed_src.unsqueeze(0),tgt.unsqueeze(0))
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    transform = Reconstruction.Transform()
    transforms = np.ones([3, 4])
    transforms[:3, :3] = transform[0].detach().cpu().numpy()
    transforms[:3, 3] = transform[1].detach().cpu().numpy()

    R_error,t_error=cal_error(transforms[:3, :3],transforms[:3, 3],gt_trans[:3,:3],gt_trans[:3,3])
    print(R_error/np.pi*180,t_error)