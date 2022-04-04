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

totalRounds = 18
'''
if training_params.method!="NoEnergy":
    rows = list(csv.reader(open(misc_variables.plotFile,'r'),delimiter=',',quoting=csv.QUOTE_NONNUMERIC))
    rows = np.squeeze(np.array(rows))
    totalRounds = len(rows)
    minRound = np.argmin(rows)
else:
    totalRounds = 1
'''

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
        reconstruction_l = reconstruction.cpu().detach().numpy().tolist()
        reconstructions.extend(reconstruction_l)
        
        reconstructionError = losses.mse(pts,reconstruction)
        samples.extend(code)
        meshNames.extend(names)
        allPts.extend(pts)
        allFaces.extend(faces)
        allNeighbors.extend(numNeighbors)
        allNeighborsMatrix.extend(neighborsMatrix)
        allWeightMatrix.extend(weightMatrix)

        meshFaces = faces[0]
        meanReconError += reconstructionError.item()
        if batchIter%1==0:
            print("global:",batchIter,len(data_classes.torch_original),meanReconError/(batchIter+1))
        batchIter+=1

    '''
    for b in range(len(samples)):
        recon = reconstructions[b]
        recon_name = meshNames[b].split(".obj")[0] + "_recon.obj"
        recon_name = os.path.join(misc_variables.outputDir,recon_name)
        print(os.path.join(misc_variables.outputDir,recon_name))
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(np.array(recon))
        objInstance.setFaces(meshFaces.detach().numpy())
        objInstance.saveAs(recon_name)

    exit()
    '''
    allPairsEnergy = 0.0
    for index,p in enumerate(pairs):
        print(rounds,index,len(pairs))
        i = p[0] ; j = p[1]
        try:
            sourceSample = samples[i]
            sourceName = meshNames[i].split(".obj")[0]
        except:
            break

        try:
            destSample = samples[j]
            destName = meshNames[j].split(".obj")[0]
        except:
            break
        interIndex=0
        pairEnergy = 0.0

        interts = np.arange(0,1.05,0.05)
        for t in interts:
            interpolatedSample = (1-t)*sourceSample + t*destSample
            interpolatedVerts = network_params.autoEncoder.decoder(torch.unsqueeze(interpolatedSample,0))
            objInstance = Obj.Obj("dummy.obj")
            outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(interIndex)+".obj"
            if not os.path.exists(misc_variables.outputDir+"p_"+str(index+1)+"/"):
                os.makedirs(misc_variables.outputDir+"p_"+str(index+1)+"/")
            outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),outName)
            if network_params.dims==2:
                z = torch.zeros(1,network_params.numPoints,1).float().cuda()
                interpolatedVerts = torch.cat((interpolatedVerts,z),2)

            objInstance.setVertices(interpolatedVerts.cpu().detach().numpy()[0])
            objInstance.setFaces(meshFaces.cpu().detach().numpy())
            objInstance.saveAs(outPath)
            interIndex+=1

            energyLoss = torch.zeros(1,1).float().cuda()
            if training_params.energy=='pdist':
                energyLoss = losses.pdist(torch.unsqueeze(allPts[i],0),interpolatedVerts,network_params.testEnergyWeight)
            if training_params.energy=='arap':
                energyLoss = losses.arap(torch.unsqueeze(allPts[i],0),interpolatedVerts,torch.unsqueeze(allNeighborsMatrix[i],0),torch.unsqueeze(allNeighbors[i],0),torch.unsqueeze(allWeightMatrix[i],0),network_params.testEnergyWeight).mean()
            if training_params.energy=='asap':
                energyLoss = losses.asap(torch.unsqueeze(allPts[i],0),interpolatedVerts,torch.unsqueeze(allNeighborsMatrix[i],0),torch.unsqueeze(allNeighbors[i],0),torch.unsqueeze(allWeightMatrix[i],0),network_params.testEnergyWeight).mean()
            pairEnergy += 0.5*energyLoss.item()
        allPairsEnergy += (pairEnergy/len(interts))
    allPairsEnergy /= len(pairs)
    print(allPairsEnergy)
