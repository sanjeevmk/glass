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
totalRounds = 15

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

#pairs = [pairs[41],pairs[42]]
minRound = training_params.test_round
if not os.path.exists(misc_variables.outputDir):
    os.makedirs(misc_variables.outputDir)

def dumpSamples(outDir,samples,faces,name):
    for i in range(len(samples)):
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(samples[i])
        objInstance.setFaces(faces)
        objInstance.saveAs(os.path.join(outDir,name+'.obj'))

radius=0.1
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

    allPts = []
    allnumNeighbors = []
    allaccnumNeighbors = []
    allneighborsMatrix = []
    allweightMatrix = []
    landmark_codes = []
    for ind,data in enumerate(data_classes.torch_full_arap_original):
        pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,names,_ = data
        pts = pts[:,:,:network_params.dims]
        pts = pts.float().cuda()
        faces = faces.int().cuda()
        numNeighbors = numNeighbors.int().cuda()
        accnumNeighbors = accnumNeighbors.int().cuda()
        neighborsMatrix = neighborsMatrix.int().cuda()
        weightMatrix = weightMatrix.float().cuda()

        code = network_params.autoEncoder.encoder(pts.transpose(2,1))
        print(names)
        print(code)

        reconstruction = network_params.autoEncoder.decoder1(code)
        reconstruction_l = reconstruction.cpu().detach().numpy().tolist()
        reconstructions.extend(reconstruction_l)
        
        reconstructionError = losses.mse(pts,reconstruction)
        samples.extend(code)
        landmark_codes.extend(code.cpu().detach().numpy())
        meshNames.append([names[0]])
        allnumNeighbors.extend(numNeighbors)
        allaccnumNeighbors.extend(accnumNeighbors)
        allneighborsMatrix.extend(neighborsMatrix)
        allweightMatrix.extend(weightMatrix)
        allPts.extend(pts)

        meshFaces = faces[0]
        meanReconError += reconstructionError.item()
        if batchIter%1==0:
            print("global:",batchIter,len(data_classes.torch_full_arap_original),meanReconError/(batchIter+1))
        batchIter+=1

    landmark_codes = np.array(landmark_codes)
    np.save('./landscape/r0_landmark_codes_noenergy.npy',landmark_codes)
    csv.writer(open('./landscape/r0_landmark_names_noenergy.csv','w'),delimiter='\n').writerows(meshNames)

    generated_codes = []
    generated_araps = []
    total_sample_num = 0
    per_shape_samples = 2000
    meshFaces = meshFaces.cpu().detach().numpy()

    for ind,data in enumerate(data_classes.torch_full_arap_original):
        print(ind)
        pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,names,_ = data
        #if names[0].startswith('2'):
        #    continue
        pts = pts[:,:,:network_params.dims]
        pts = pts.float().cuda()
        faces = faces.int().cuda()
        numNeighbors = numNeighbors.int().cuda()
        accnumNeighbors = accnumNeighbors.int().cuda()
        neighborsMatrix = neighborsMatrix.int().cuda()
        weightMatrix = weightMatrix.float().cuda()

        code = network_params.autoEncoder.encoder(pts.transpose(2,1))

        for sample_num in range(per_shape_samples):
            print(ind,sample_num)
            network_params.noise.normal_()
            rand_normal = Variable(network_params.noise[:code.size()[0],:])
            rand_sample = code+rand_normal
            generated_codes.extend(rand_sample.cpu().detach().numpy())

            reconstruction = network_params.autoEncoder.decoder1(rand_sample)
            energyLoss,_ = losses.arap(pts,reconstruction,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
            generated_araps.append(energyLoss.mean().item())
            name = str(total_sample_num + sample_num)
            dumpSamples(misc_variables.outputDir,reconstruction.cpu().detach().numpy(),meshFaces,name)
        total_sample_num += per_shape_samples
    
    generated_codes = np.array(generated_codes)
    generated_araps = np.array(generated_araps)

    np.save('./landscape/r0_random_codes_noenergy.npy',generated_codes)
    np.save('./landscape/r0_random_arap_noenergy.npy',generated_araps)
