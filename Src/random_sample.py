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

def dumpSamples(outDir,samples,faces,name):
    for i in range(len(samples)):
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(samples[i])
        objInstance.setFaces(faces)
        objInstance.saveAs(os.path.join(outDir,name+'.obj'))

#pairs = [pairs[41],pairs[42]]
minRound = training_params.test_round
if not os.path.exists(misc_variables.randoutputDir):
    os.makedirs(misc_variables.randoutputDir)

radius=0.1
extrapolation_dir = misc_variables.outputDir.replace("interpolation","extrapolation")
extrapolation_dir_test = misc_variables.outputDir.replace("interpolation","extrapolation_test")

for rounds in range(totalRounds):
    #if training_params.method!="NoEnergy":
    if rounds != minRound:
        continue
    print(rounds)
    network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))
    network_params.autoEncoder.eval()
    samples = []
    meshNames = []
    meshFaces = []
    reconstructions = []
    meanReconError=0
    batchIter=0

    for ind,data in enumerate(data_classes.torch_original):
        pts,faces,names = data
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
        samples.extend(code)
        meshNames.extend(names)

        meshFaces = faces[0]
        meanReconError += reconstructionError.item()
        if batchIter%1==0:
            print("global:",batchIter,len(data_classes.torch_full_arap_original),meanReconError/(batchIter+1))
        batchIter+=1

    per_shape_samples = 500
    total_sample_num = 0

    for ind,data in enumerate(data_classes.torch_original):
        print(ind)
        pts,faces,name = data
        pts = pts[:,:,:network_params.dims]
        pts = pts.float().cuda()
        faces = faces.int().cuda()

        code = network_params.autoEncoder.encoder(pts.transpose(2,1))

        for sample_num in range(per_shape_samples):
            print(ind,sample_num,total_sample_num)
            network_params.noise.normal_()
            rand_normal = Variable(0.5*network_params.noise[:code.size()[0],:])
            rand_sample = code+rand_normal

            reconstruction = network_params.autoEncoder.decoder1(rand_sample)
            name = str(total_sample_num)
            dumpSamples(misc_variables.randoutputDir,reconstruction.cpu().detach().numpy(),meshFaces.cpu().detach().numpy(),name)
            total_sample_num +=1
