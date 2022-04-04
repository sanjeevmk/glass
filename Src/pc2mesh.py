import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import sys

import numpy as np
import torch.optim as optim
import torch.nn as nn
import time
from sklearn.neighbors import NearestNeighbors
import trimesh
import torch
import os

from torch.autograd import Variable
import header
import init
import sys
from train import trainVAE

import csv 
#import matplotlib.pyplot as plt
from Utils import writer,Off,Obj,rect_remesh
import numpy as np
import os


neigh = NearestNeighbors(1, 0.4)

count = 0

def normalize(vertices):
    bmin = np.min(vertices,axis=0)
    bmax = np.max(vertices,axis=0)
    bcenter = (bmin+bmax)/2.0
    vertices -= bcenter
    bmin = np.min(vertices,axis=0)
    bmax = np.max(vertices,axis=0)
    diagsq = np.sum(np.power(bmax-bmin,2))
    diag = np.sqrt(diagsq)
    s = np.eye(3)
    s *= (1.0/diag)
    normed_vertices = np.dot(vertices,s)

    return normed_vertices

def compute_correspondences(inputA,network_params,misc_variables,losses,faces):
    source = trimesh.load(inputA, process=False)
    source_vertices = normalize(source.vertices)
    source_verts = torch.unsqueeze(torch.from_numpy(np.array(source_vertices)).float().cuda(),0)

    network_params.corrEncoder.eval()
    for parameter in network_params.autoEncoder.decoder1.parameters():
        parameter.requires_grad = False
    for parameter in network_params.autoEncoder.encoder.parameters():
        parameter.requires_grad = False
    for parameter in network_params.corrEncoder.parameters():
        parameter.requires_grad = True 

    bestError =1e9
    bestReconstruction = None
    for attempt in range(50):
        network_params.corroptimizer.zero_grad()
        s_code = network_params.corrEncoder(source_verts.transpose(2,1))
        reconstructed = network_params.autoEncoder.decoder1(s_code)
        d1,d2 = losses.cd(source_verts,reconstructed)
        L = d1.mean() + d2.mean()
        if L.item() < bestError:
            bestError = L.item()
            bestReconstruction = reconstructed[0].cpu().detach().numpy()
            objInstance = Obj.Obj("dummy.obj")
            objInstance.setVertices(bestReconstruction)
            objInstance.setFaces(faces)
            fname = inputA.split(".ply")[0].split("/")[-1]+".obj"
            objInstance.saveAs(os.path.join(misc_variables.reconDir,fname))
        L.backward()
        network_params.corroptimizer.step()
        print(inputA,attempt,L.item(),bestError)
    #s_code = network_params.corrEncoder(source_verts.transpose(2,1))
    #source_reconstructed = network_params.autoEncoder.decoder1(s_code)[0].cpu().detach().numpy()
    objInstance = Obj.Obj("dummy.obj")
    objInstance.setVertices(bestReconstruction)
    objInstance.setFaces(faces)
    fname = inputA.split(".ply")[0].split("/")[-1]+".obj"
    objInstance.saveAs(os.path.join(misc_variables.reconDir,fname))
    

def main(config,src_mesh_file,template_obj_file):
    training_params,data_classes,network_params,misc_variables,losses = init.initialize(config)
    if not os.path.exists(misc_variables.outputDir):
        os.makedirs(misc_variables.outputDir)
    #network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(training_params.target_round)))
    network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r5"))
    network_params.autoEncoder.eval()
    network_params.corrEncoder.load_state_dict(torch.load(network_params.corrweightPath+"_r"+str(training_params.target_round)))
    network_params.corrEncoder.eval()
    objInstance = Obj.Obj(template_obj_file)
    objInstance.readContents()
    p,f = objInstance.readPointsAndFaces(normed=True)
    #compute_correspondences(inputA=os.path.join(misc_variables.test_corr_root, src_mesh_file),network_params=network_params,misc_variables=misc_variables,losses=losses,faces=f)
    compute_correspondences(inputA=src_mesh_file,network_params=network_params,misc_variables=misc_variables,losses=losses,faces=f)

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3])
