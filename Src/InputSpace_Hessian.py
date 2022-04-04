import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs

import Energies
import Leapfrog
import Hessian 
from Utils import writer,Off,Obj,Pts,rect_remesh
import header 
from torch.autograd import Variable
import math

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, create_graph=create_graph,retain_graph=True)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    jacobian_mat = torch.stack(jac).reshape(y.shape + x.shape)
    return torch.squeeze(jacobian_mat)

def jacobian_hessian(y, x):
    jac = jacobian(y, x, create_graph=True)
    hess = jacobian(jac,x)
    return jac,hess

def hessianExplore(training_params,data_classes,network_params,losses,misc_variables,rounds):
    shapes = []
    meshNames = []
    for ind,data in enumerate(data_classes.torch_original):
        pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,area,names,p_pts = data
        pts = pts[:,:,:network_params.dims]
        pts = pts.float().cuda()
        p_pts = p_pts[:,:,:network_params.dims]
        p_pts = p_pts.float().cuda()
        area = area.float().cuda()
        faces = faces.int().cuda()
        numNeighbors = numNeighbors.int().cuda()
        accnumNeighbors = accnumNeighbors.int().cuda()
        neighborsMatrix = neighborsMatrix.int().cuda()
        weightMatrix = weightMatrix.float().cuda()

        proposed_shape = pts.clone().detach().requires_grad_(True)
        energy,_ = losses.arap(pts,proposed_shape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
        meanEnergy = torch.mean(torch.mean(energy,2),1)

        grad,hess= jacobian_hessian(meanEnergy,proposed_shape)
