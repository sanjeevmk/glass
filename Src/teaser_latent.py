import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
from torch.autograd import Variable
import header
import init
import sys
import train_parental
import randomSample
import latentSpaceExplore_VanillaHMC,latentSpaceExplore_NUTSHmc
config = sys.argv[1]
training_params,data_classes,network_params,misc_variables,losses = init.initialize(config)
from train import trainVAE
import high_res_projection
import csv 
#import matplotlib.pyplot as plt
from Utils import writer,Off,Obj,rect_remesh
import numpy as np
import os
import trimesh
import re
totalRounds = 15


for rounds in range(totalRounds):
    #if training_params.method!="NoEnergy":
    if rounds != training_params.test_round:
        continue
    print(rounds)
    network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))
    network_params.autoEncoder.eval()
    samples = []
    meshNames = []
    reconstructions = []
    meanReconError=0
    batchIter=0

    for ind,data in enumerate(data_classes.torch_original):
        pts,faces,names = data
        if not (names[0].startswith('tr') or names[0].startswith('0') or names[0].startswith('2') or names[0].startswith('3') or names[0].startswith('4') or names[0].startswith('5') or names[0].startswith('6')):
            continue

        pts = pts[:,:,:network_params.dims]
        pts = pts.float().cuda()
        faces = faces.int().cuda()

        code = network_params.autoEncoder.encoder(pts.transpose(2,1))
        print(names)
        print(code)

        reconstruction = network_params.autoEncoder.decoder1(code)
        reconstruction_l = reconstruction.cpu().detach().numpy().tolist()
        reconstructions.extend(reconstruction_l)
        
        reconstructionError = losses.mse(pts,reconstruction)
        samples.extend(code.cpu().detach().numpy())
        meshNames.extend(list(names))

        meshFaces = faces[0]
        meanReconError += reconstructionError.item()
        if batchIter%1==0:
            print("global:",batchIter,len(data_classes.torch_full_arap_original),meanReconError/(batchIter+1))
        batchIter+=1

    samples = np.array(samples)
    meshNames = [[x] for x in meshNames]
    print(samples.shape)
    np.save('./tsne_rounds/faust10_teaser_6.npy',samples)
    csv.writer(open('./tsne_rounds/faust10_names_6.txt','w'),delimiter=',').writerows(meshNames)
