import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
from torch.autograd import Variable
import header
import init
import sys
import train
import randomSample
import latentSpaceExplore_VanillaHMC,latentSpaceExplore_NUTSHmc,latentSpaceExplore_PerturbOptimize
config = sys.argv[1]
training_params,data_classes,network_params,misc_variables,losses = init.initialize(config)
from train import trainVAE

import csv 
import matplotlib.pyplot as plt
from Utils import writer,Off,Obj
import numpy as np
import os
import Energies

totalRounds = -1
if training_params.method!="NoEnergy":
    rows = list(csv.reader(open(misc_variables.plotFile,'r'),delimiter=',',quoting=csv.QUOTE_NONNUMERIC))
    rows = np.squeeze(np.array(rows))
    totalRounds = len(rows)
    minRound = np.argmin(rows)
else:
    totalRounds = 1

datalen = len(data_classes.original)
pairs = []
for i in range(datalen):
    for j in range(datalen):
        if i==j:
            continue
        if (j,i) in pairs:
            continue
        pairs.append((i,j))

minRound =4
source_shape = "carpet.obj"
for rounds in range(totalRounds):
    if training_params.method!="NoEnergy":
        if rounds != minRound:
            continue
    network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))
    network_params.autoEncoder.eval()


    def dumpSamples(outDir,samples):
        for i in range(len(samples)):
            objInstance = Obj.Obj("dummy.obj")
            objInstance.setVertices(samples[i])
            objInstance.setFaces(faces)
            objInstance.saveAs(os.path.join(outDir,str(rounds) + "_" + names.split(".")[0] + "_" + str(i)+".obj"))

    for ind,data in enumerate(data_classes.original):
        samples = []
        energies = []
        print("HMC Source Data:",ind,len(data_classes.original),flush=True)
        pts,faces,numNeighbors,neighborsMatrix,weightMatrix,area,names = data
        print(names)

        if source_shape not in names:
            continue

        query = np.expand_dims(pts[:,:network_params.dims],0)
        query = np.reshape(query,(1,-1))
        pts = torch.unsqueeze(torch.from_numpy(pts).float().cuda(),0)
        pts = pts[:,:,:network_params.dims]
        numNeighbors = torch.unsqueeze(torch.from_numpy(numNeighbors).int().cuda(),0)
        neighborsMatrix = torch.unsqueeze(torch.from_numpy(neighborsMatrix).int().cuda(),0)
        weightMatrix = torch.unsqueeze(torch.from_numpy(weightMatrix).float().cuda(),0)
        area = torch.unsqueeze(torch.from_numpy(area).float().cuda(),0)

        energy_fn = None

        if training_params.energy == 'pdist':
            energy_fn = Energies.LogPairwiseDistanceEnergy(pts,network_params.testEnergyWeight,network_params.autoEncoder.decoder)
        #arapEnergy = ArapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,testEnergyWeight)
        #arapEnergy = CArapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,alpha,area,testEnergyWeight)
        code = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))

        burnin=0
        stepSize  = 0.001
        codeSamples,energy,_ = header.nuts6(stepSize,energy_fn.forward,1000,burnin,code.detach().cpu().numpy()[0],0.99)
        if len(codeSamples)<2:
            continue
        codeSamples = codeSamples[1:,:]
        energy = energy[1:]
        energy *= -1
        energy /= network_params.testEnergyWeight.item()
        print(len(codeSamples),np.min(energy))
        if len(codeSamples)==0:
            continue
        codeSamples = torch.tensor(codeSamples,device='cuda',dtype=torch.float)

        samples = torch.zeros((1,network_params.numPoints,network_params.dims)).float().cuda(0)

        for bind in range(0,len(codeSamples),training_params.batch):
            shapeSamples = network_params.autoEncoder.decoder(codeSamples[bind:bind+training_params.batch,:])
            samples = torch.cat([samples,shapeSamples],0)
        
        samples = samples[1:,:,:]
        samples = samples.detach().cpu().numpy()

        samples = np.array(samples)

        if not os.path.exists(misc_variables.outputDir+"hmc_samples/"):
            os.makedirs(misc_variables.outputDir+"hmc_samples/")
        samples = np.reshape(samples,(samples.shape[0],network_params.numPoints,network_params.dims))
        if network_params.dims==2:
            z = np.zeros((samples.shape[0],network_params.numPoints,1))
            samples = np.concatenate((samples,z),2)
        dumpSamples(misc_variables.outputDir+"hmc_samples",samples)