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
if not os.path.exists(misc_variables.randoutputDir):
    os.makedirs(misc_variables.randoutputDir)

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
        meshNames.extend(names)
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

    random_samples = []
    sample_smoothness_table = []
    sample_coverage_table = [] 

    if training_params.sampling:
        _,template_face,_ = data_classes.original[0]

        samplingRounds=training_params.num_samples
        for i in range(samplingRounds):
            if i%500==0:
                print(i,samplingRounds)
            rand_sample = 0.1*torch.randn(1,network_params.bottleneck).float().cuda()
            rand_out = network_params.autoEncoder.decoder1(rand_sample)
            if network_params.dims==2:
                z = torch.zeros(1,network_params.numPoints,1).float().cuda()
                rand_out = torch.cat((rand_out,z),2)

            if not os.path.exists(misc_variables.randoutputDir):
                os.makedirs(misc_variables.randoutputDir)
            outPath = os.path.join(misc_variables.randoutputDir,str(i)+".obj")
            objInstance = Obj.Obj("dummy.obj")
            objInstance.setVertices(rand_out.cpu().detach().numpy()[0])
            objInstance.setFaces(template_face)
            objInstance.saveAs(outPath)

            random_samples.append(rand_out.cpu().detach().numpy()[0])

    if training_params.sampling:
        random_samples = np.array(random_samples)
        for idx in range(len(random_samples)):
            vertices = random_samples[idx]
            bmin = np.min(vertices,0)
            bmax = np.max(vertices,0)
            bcenter = (bmin+bmax)/2.0
            vertices -= bcenter
            bmin = np.min(vertices,axis=0)
            bmax = np.max(vertices,axis=0)
            diagsq = np.sum(np.power(bmax-bmin,2))
            diag = np.sqrt(diagsq)
            s = np.eye(3)
            s *= (1.0/diag)
            normed_vertices = np.dot(vertices,s)
            random_samples[idx] = normed_vertices
    else:
        if data_classes.test_set:
            sample_files = os.listdir(misc_variables.sampleDir)
            for i,f in enumerate(sample_files):
                if i%500==0:
                    print(i,len(sample_files))
                if not f.endswith(".obj"):
                    continue
                excMesh = trimesh.load(os.path.join(misc_variables.sampleDir,f),process=False)
                vertices = np.array(excMesh.vertices)
                bmin = np.min(vertices,0)
                bmax = np.max(vertices,0)
                bcenter = (bmin+bmax)/2.0
                vertices -= bcenter
                bmin = np.min(vertices,axis=0)
                bmax = np.max(vertices,axis=0)
                diagsq = np.sum(np.power(bmax-bmin,2))
                diag = np.sqrt(diagsq)
                s = np.eye(3)
                s *= (1.0/diag)
                normed_vertices = np.dot(vertices,s)
                random_samples.append(normed_vertices)

    test_set_vertices = []
    if data_classes.test_set:
        test_files = os.listdir(data_classes.test_set)
        for f in test_files:
            if not f.endswith(".obj"):
                continue
            excMesh = trimesh.load(os.path.join(data_classes.test_set,f),process=False)
            vertices = np.array(excMesh.vertices)
            bmin = np.min(vertices,0)
            bmax = np.max(vertices,0)
            bcenter = (bmin+bmax)/2.0
            vertices -= bcenter
            bmin = np.min(vertices,axis=0)
            bmax = np.max(vertices,axis=0)
            diagsq = np.sum(np.power(bmax-bmin,2))
            diag = np.sqrt(diagsq)
            s = np.eye(3)
            s *= (1.0/diag)
            normed_vertices = np.dot(vertices,s)
            test_set_vertices.append(normed_vertices)

    minDistances = []
    if len(random_samples)>0:
        random_samples = np.array(random_samples)
        test_set_vertices = np.array(test_set_vertices)

        print(random_samples.shape,test_set_vertices.shape)

        for i in range(test_set_vertices.shape[0]):
            excvert = np.expand_dims(test_set_vertices[i,:,:],0)
            dist = (random_samples-excvert)**2
            dist = np.sqrt(np.sum(dist,2))
            dist = np.mean(dist,1)
            minDist = np.min(dist)
            minIndex = np.argmin(dist)
            #print(excNames[i],genNames[minIndex])
            minDistances.append(minDist)

        print(np.mean(minDistances))
        sample_coverage_table.append(np.mean(minDistances))

    outPath = os.path.join(misc_variables.randoutputDir,"sample_smoothness.txt")
    if len(sample_smoothness_table)>0:
        with open(outPath,'w') as f:
            f.write(str(np.mean(sample_smoothness_table))+'\n')

    outPath = os.path.join(misc_variables.randoutputDir,"sample_dist.txt")
    if len(sample_coverage_table)>0:
        with open(outPath,'w') as f:
            f.write(str(np.mean(sample_coverage_table))+'\n')
