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

import csv 
#import matplotlib.pyplot as plt
from Utils import writer,Off,Obj,rect_remesh
import numpy as np
import os

totalRounds = 1
'''
if training_params.method!="NoEnergy":
    rows = list(csv.reader(open(misc_variables.plotFile,'r'),delimiter=',',quoting=csv.QUOTE_NONNUMERIC))
    rows = np.squeeze(np.array(rows))
    totalRounds = len(rows)
    minRound = np.argmin(rows)
else:
    totalRounds = 1

datalen = len(data_classes.original)
pairs = []
import random
for i in range(datalen):
    for j in range(datalen):
        if i==j:
            continue
        if (j,i) in pairs:
            continue
        options = [x for x in range(datalen) if x!=i and x!=j]
        #third = random.choice(options)
        pairs.append((i,j)) #,third))
'''

minRound = 7
if not os.path.exists(misc_variables.outputDir):
    os.makedirs(misc_variables.outputDir)
if not os.path.exists(misc_variables.randoutputDir):
    os.makedirs(misc_variables.randoutputDir)

#if training_params.method!="NoEnergy":
#    if rounds != minRound:
#        continue
network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_"+misc_variables.featureFile.split(".")[0]))
network_params.autoEncoder.eval()

samples = []
meshNames = []
meshFaces = []
reconstructions = []
meanReconError=0
batchIter=0
for ind,data in enumerate(data_classes.torch_original):
    pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,area,names,p_pts = data
    pts = pts[:,:,:network_params.dims]
    pts = pts.float().cuda()
    code = network_params.autoEncoder.encoder(pts.transpose(2,1))
    reconstruction = network_params.autoEncoder.decoder1(code)
    reconstruction_l = reconstruction.cpu().detach().numpy().tolist()
    
    reconstructionError = losses.mse(pts,reconstruction)
    samples.extend(code.detach().cpu().numpy().tolist())
    meshNames.extend(names)
    meshFaces = faces[0]
    meanReconError += reconstructionError.item()
    if batchIter%1==0:
        print("global:",batchIter,len(data_classes.torch_original),meanReconError/(batchIter+1))
    batchIter+=1

import pickle
outDict = {}
for i in range(len(meshNames)):
    outDict[meshNames[i]] = samples[i]

with open(os.path.join(misc_variables.outputDir,misc_variables.featureFile),"wb") as f:
    pickle.dump(outDict,f)
