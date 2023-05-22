import os
#os.environ['CUDA_VISIBLE_DEVICES']='2'
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F

import open3d as o3d

from model import build_grid,MLP_Decoder,setup_seed
from loss import CLGD
from tqdm import tqdm

def df(x, wrt):
    B, M = x.shape
    return ag.grad(x.flatten(), wrt,
                    grad_outputs=torch.ones(B * M , dtype=torch.float32).
                    to(x.device), create_graph=True)[0]

def gradient(xyz,uv):
    x,y,z=xyz[...,0],xyz[...,1],xyz[...,2]
    dx=df(x,uv).unsqueeze(2)     #(B,N,1,2)
    dy=df(y,uv).unsqueeze(2)
    dz=df(z,uv).unsqueeze(2)

    dxyz=torch.cat((dx,dy,dz),dim=2)    #(B,N,3,2)
    return dxyz


def write_ply(save_path,points,normals=None):
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals=o3d.utility.Vector3dVector(normals)
    
    o3d.io.write_point_cloud(save_path,pcd)

def shape_rec(gt_pc:np.array,save_path:str,n_epoch=10000,lr=1e-3):
    gt_pc=torch.from_numpy(gt_pc).cuda().float().unsqueeze(0)
    decoder=MLP_Decoder().cuda()
    grid=build_grid(64)    #(10000,2)
    loss_func=CLGD(weighted_query=False)
    grid=torch.from_numpy(grid).cuda().float().unsqueeze(0)
    optimizer=torch.optim.Adam(decoder.parameters(),lr=lr,weight_decay=1e-5)
    grid.requires_grad=True
    for i in tqdm(range(1,n_epoch+1)):
        decoder.train()
        rec_pc=decoder(grid)
        loss=loss_func(rec_pc,gt_pc) #*1000

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%1000==0:
            decoder.eval()
            rec_pc=decoder(grid)
            grad=gradient(rec_pc,grid)
            
            grad_u,grad_v=grad[...,0],grad[...,1]

            normal=torch.cross(grad_u,grad_v)
            normal=F.normalize(normal,dim=-1) 

            write_ply(os.path.join(save_path,'%d.ply'%i),rec_pc[0].cpu().detach().numpy(),normal[0].cpu().detach().numpy())

    torch.save(decoder.state_dict(),os.path.join(save_path,'model.pth'))

if __name__=='__main__':
    setup_seed(1)
    gt_pc=o3d.io.read_point_cloud(os.path.join('example','shape_reconstruction','samples.ply'))
    gt_pc=np.array(gt_pc.points).astype(np.float32)
    shape_rec(gt_pc,os.path.join('example','shape_reconstruction'))
