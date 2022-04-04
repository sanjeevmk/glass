import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
from torch.autograd import Variable
import header
import init
import sys
import train_parental
import randomSample
import latentSpaceExplore_VanillaHMC,latentSpaceExplore_NUTSHmc,latentSpaceExplore_PerturbOptimize
config = sys.argv[1]
training_params,data_classes,network_params,misc_variables,losses = init.initialize(config)
from train import trainVAE

import csv 
#import matplotlib.pyplot as plt
from Utils import writer,Off,Obj,rect_remesh
import numpy as np
import os

totalRounds = 1

datalen = len(data_classes.original)
pairs = []
import random
for i in range(datalen):
    for j in range(datalen):
        if i==j:
            continue
        #if (j,i) in pairs:
        #    continue
        #options = [x for x in range(datalen) if x!=i and x!=j]
        #third = random.choice(options)
        pairs.append((i,j)) #,third))

minRound = 0
print("here")
if not os.path.exists(misc_variables.outputDir):
    os.makedirs(misc_variables.outputDir)
for rounds in range(totalRounds):
    #if training_params.method!="NoEnergy":
    #    if rounds != minRound:
    #        continue
    print(rounds)
    network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))
    network_params.autoEncoder.eval()
    samples = []
    allPts = []
    allFaces = []
    allNeighbors = []
    allNeighborsMatrix = []
    allWeightMatrix = []
    allArea = []

    meshNames = []
    meshFaces = []
    reconstructions = []
    meanReconError=0
    batchIter=0
    allSamples = []
    allNames = []
    allEnergies = []
    for ind,data in enumerate(data_classes.torch_original):
        pts,faces,numNeighbors,neighborsMatrix,weightMatrix,area,names,p_pts = data
        pts = pts[:,:,:network_params.dims]
        pts = pts.float().cuda()
        p_pts = p_pts[:,:,:network_params.dims]
        p_pts = p_pts.float().cuda()
        area = area.float().cuda()
        faces = faces.int().cuda()
        numNeighbors = numNeighbors.int().cuda()
        neighborsMatrix = neighborsMatrix.int().cuda()
        weightMatrix = weightMatrix.float().cuda()

        #code = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
        code = network_params.autoEncoder.encoder(pts.transpose(2,1))
        #print(names)
        #print(code)
        #network_params.noise.normal_()
        #addedNoise = Variable(0.01*network_params.noise[:code.size()[0],:])
        #code += addedNoise
        reconstruction = network_params.autoEncoder.decoder(code)

        if training_params.energy=='pdist':
            energyLoss = losses.pdist(pts,reconstruction,network_params.energyWeight)
        if training_params.energy=='arap':
            energyLoss = losses.arap(pts,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,network_params.energyWeight)
        if training_params.energy=='arap2d':
            energyLoss = losses.arap2d(pts,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,network_params.energyWeight)
        if training_params.energy=='asap':
            energyLoss = losses.asap(pts,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,network_params.energyWeight)
        if training_params.energy=='asap2d':
            energyLoss = losses.asap2d(pts,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,network_params.energyWeight)

        samples.extend(code)
        meshNames.extend(names)
        print(code.cpu().detach().numpy().shape)
        allSamples.extend(code.cpu().detach().numpy())
        allNames.append([names[0].split('.')[0]+"_original"])
        allEnergies.append([energyLoss.mean().item()])
        radii = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]
        print(ind)
        for r in radii:
            print(r)
            for j in range(100):
                network_params.noise.normal_()
                addedNoise = Variable(r*network_params.noise[:code.size()[0],:])
                codeSample = code + addedNoise
                reconstruction = network_params.autoEncoder.decoder(codeSample)
                energyLoss = losses.arap(pts,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,network_params.energyWeight)
                allSamples.extend(codeSample.cpu().detach().numpy())
                newname = names[0].split('.')[0] + "_" + str(r) + "_" + str(j)
                allNames.append([newname])
                allEnergies.append([energyLoss.mean().item()])


        meshFaces = faces[0]
        batchIter+=1
    
    np.save('arap_codes.npy',np.array(allSamples))
    csv.writer(open('arap_names.csv','w'),delimiter='\n').writerows(allNames)
    csv.writer(open('arap_energies.csv','w'),delimiter='\n').writerows(allEnergies)
