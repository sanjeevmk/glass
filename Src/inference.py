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
'''

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

minRound = 7
if not os.path.exists(misc_variables.outputDir):
    os.makedirs(misc_variables.outputDir)
if not os.path.exists(misc_variables.randoutputDir):
    os.makedirs(misc_variables.randoutputDir)

sampleRound=7
network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(sampleRound)))
network_params.autoEncoder.eval()
for ind,data in enumerate(data_classes.torch_original):
    pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,area,names,p_pts = data
    meshFaces = faces[0]
    break
import json
jsonArgs = json.loads(open(config,'r').read())
prefix = jsonArgs['scaperoot'].split("/")[-2]


meshFaces = []
samples = []
batchIter=0

for ind,data in enumerate(data_classes.torch_original):
    pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,area,names,p_pts = data
    pts = pts[:,:,:network_params.dims]
    pts = pts.float().cuda()
    code = network_params.autoEncoder.encoder(pts.transpose(2,1))
    samples.extend(code)
    meshFaces = faces[0]
    if batchIter%1==0:
        print("global:",batchIter,len(data_classes.torch_original))
    batchIter+=1

#for ind,data in enumerate(data_classes.torch_original):
#    pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,area,names,p_pts = data
#    meshFaces = faces[0]
#    break
samplingRounds=1000
for j in range(len(samples)):
    print(j,len(samples))
    for i in range(samplingRounds):
        rand_sample = 0.3*torch.randn(1,network_params.bottleneck).float().cuda()
        sourceSample = torch.unsqueeze(samples[j],0)
        rand_sample = (sourceSample + rand_sample)
        rand_out = network_params.autoEncoder.decoder1(rand_sample)
        if network_params.dims==2:
            z = torch.zeros(1,network_params.numPoints,1).float().cuda()
            rand_out = torch.cat((rand_out,z),2)

        if not os.path.exists(misc_variables.randoutputDir):
            os.makedirs(misc_variables.randoutputDir)
        outPath = os.path.join(misc_variables.randoutputDir,prefix+'_'+str(j*samplingRounds+i)+".obj")
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(rand_out.cpu().detach().numpy()[0])
        objInstance.setFaces(meshFaces.detach().numpy())
        objInstance.saveAs(outPath)

exit()
samplingRounds=4000
for i in range(samplingRounds):
    if i%500==0:
        print(i,samplingRounds)
    rand_sample = 0.2*torch.randn(1,network_params.bottleneck).float().cuda()
    rand_out = network_params.autoEncoder.decoder1(rand_sample)
    if network_params.dims==2:
        z = torch.zeros(1,network_params.numPoints,1).float().cuda()
        rand_out = torch.cat((rand_out,z),2)

    if not os.path.exists(misc_variables.randoutputDir):
        os.makedirs(misc_variables.randoutputDir)
    outPath = os.path.join(misc_variables.randoutputDir,prefix+'_'+str(i)+".obj")
    objInstance = Obj.Obj("dummy.obj")
    objInstance.setVertices(rand_out.cpu().detach().numpy()[0])
    objInstance.setFaces(meshFaces.detach().numpy())
    objInstance.saveAs(outPath)

exit()
for rounds in range(totalRounds):
    #if training_params.method!="NoEnergy":
    #    if rounds != minRound:
    #        continue
    network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))
    network_params.autoEncoder.eval()

    '''
    meshFaces = []
    samples = []
    batchIter=0

    for ind,data in enumerate(data_classes.torch_original):
        pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,area,names,p_pts = data
        pts = pts[:,:,:network_params.dims]
        pts = pts.float().cuda()
        code = network_params.autoEncoder.encoder(pts.transpose(2,1))
        samples.extend(code)
        meshFaces = faces[0]
        if batchIter%1==0:
            print("global:",batchIter,len(data_classes.torch_original))
        batchIter+=1

    #for ind,data in enumerate(data_classes.torch_original):
    #    pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,area,names,p_pts = data
    #    meshFaces = faces[0]
    #    break
    samplingRounds=500
    for j in range(len(samples)):
        print(j,len(samples))
        for i in range(samplingRounds):
            rand_sample = 0.3*torch.randn(1,network_params.bottleneck).float().cuda()
            sourceSample = torch.unsqueeze(samples[j],0)
            rand_sample = (sourceSample + rand_sample)
            rand_out = network_params.autoEncoder.decoder1(rand_sample)
            if network_params.dims==2:
                z = torch.zeros(1,network_params.numPoints,1).float().cuda()
                rand_out = torch.cat((rand_out,z),2)

            if not os.path.exists(misc_variables.randoutputDir):
                os.makedirs(misc_variables.randoutputDir)
            outPath = os.path.join(misc_variables.randoutputDir,str(j*samplingRounds+i)+".obj")
            objInstance = Obj.Obj("dummy.obj")
            objInstance.setVertices(rand_out.cpu().detach().numpy()[0])
            objInstance.setFaces(meshFaces.detach().numpy())
            objInstance.saveAs(outPath)
    '''
    print(rounds)

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
        '''
        p_pts = p_pts[:,:,:network_params.dims]
        p_pts = p_pts.float().cuda()
        area = area.float().cuda()
        faces = faces.int().cuda()
        numNeighbors = numNeighbors.int().cuda()
        neighborsMatrix = neighborsMatrix.int().cuda()
        weightMatrix = weightMatrix.float().cuda()
        '''
        #code = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
        code = network_params.autoEncoder.encoder(pts.transpose(2,1))
        print(names)
        print(code)
        network_params.noise.normal_()
        addedNoise = Variable(0.01*network_params.noise[:code.size()[0],:])
        #code += addedNoise
        reconstruction = network_params.autoEncoder.decoder1(code)
        reconstruction_l = reconstruction.cpu().detach().numpy().tolist()
        reconstructions.extend(reconstruction_l)
        
        reconstructionError = losses.mse(pts,reconstruction)
        samples.extend(code)
        meshNames.extend(names)
        meshFaces = faces[0]
        meanReconError += reconstructionError.item()
        if batchIter%1==0:
            print("global:",batchIter,len(data_classes.torch_original),meanReconError/(batchIter+1))
        batchIter+=1

    for b in range(len(samples)):
        recon = reconstructions[b]
        recon_name = meshNames[b].split(".obj")[0] + "_recon.obj"
        recon_name = os.path.join(misc_variables.outputDir,recon_name)
        print(os.path.join(misc_variables.outputDir,recon_name))
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(np.array(recon))
        objInstance.setFaces(meshFaces.detach().numpy())
        objInstance.saveAs(recon_name)

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

        for t in np.arange(0,1.05,0.05):
            interpolatedSample = (1-t)*sourceSample + t*destSample
            interpolatedVerts = network_params.autoEncoder.decoder1(torch.unsqueeze(interpolatedSample,0))
            objInstance = Obj.Obj("dummy.obj")
            outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(interIndex)+".obj"
            if not os.path.exists(misc_variables.outputDir+"p_"+str(index+1)+"/"):
                os.makedirs(misc_variables.outputDir+"p_"+str(index+1)+"/")
            outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),outName)
            if network_params.dims==2:
                z = torch.zeros(1,network_params.numPoints,1).float().cuda()
                interpolatedVerts = torch.cat((interpolatedVerts,z),2)

            objInstance.setVertices(interpolatedVerts.cpu().detach().numpy()[0])
            objInstance.setFaces(meshFaces.detach().numpy())
            objInstance.saveAs(outPath)
            interIndex+=1


    '''
    for index,p in enumerate(pairs):
        print(rounds,index,len(pairs))
        i = p[0] ; j = p[1] ; k = p[2]
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

        try:
            destSample2 = samples[k]
            destName2 = meshNames[k].split(".obj")[0]
        except:
            break

        interIndex=0

        interpolatedSample = (sourceSample + destSample + destSample2)/3.0
        interpolatedVerts = network_params.autoEncoder.decoder(torch.unsqueeze(interpolatedSample,0))
        objInstance = Obj.Obj("dummy.obj")
        outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + destName2 + "_" + str(interIndex)+".obj"
        if not os.path.exists(misc_variables.outputDir+"p_"+str(index+1)+"/"):
            os.makedirs(misc_variables.outputDir+"p_"+str(index+1)+"/")
        outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),outName)
        if network_params.dims==2:
            z = torch.zeros(1,network_params.numPoints,1).float().cuda()
            interpolatedVerts = torch.cat((interpolatedVerts,z),2)

        objInstance.setVertices(interpolatedVerts.cpu().detach().numpy()[0])
        objInstance.setFaces(meshFaces.detach().numpy())
        objInstance.saveAs(outPath)
        interIndex+=1
    samplingRounds=100
    for index,p in enumerate(pairs):
        print(rounds,index,len(pairs))
        i = p[0] 
        try:
            sourceSample = samples[i]
            sourceName = meshNames[i].split(".obj")[0]
        except:
            break

        for i in range(samplingRounds):
            rand_sample = 0.3*torch.randn(1,network_params.bottleneck).float().cuda()
            rand_sample = (sourceSample + rand_sample)
            rand_out = network_params.autoEncoder.decoder(rand_sample)
            if network_params.dims==2:
                z = torch.zeros(1,network_params.numPoints,1).float().cuda()
                rand_out = torch.cat((rand_out,z),2)

            if not os.path.exists(misc_variables.outputDir+"random_samples/"):
                os.makedirs(misc_variables.outputDir+"random_samples/")
            outPath = os.path.join(misc_variables.outputDir+"random_samples/",str(i)+".obj")
            objInstance = Obj.Obj("dummy.obj")
            objInstance.setVertices(rand_out.cpu().detach().numpy()[0])
            objInstance.setFaces(meshFaces.detach().numpy())
            objInstance.saveAs(outPath)
    '''
    '''
    samplingRounds=100
    samplesTensor = torch.cat(samples,0)
    for i in range(samplingRounds):
        if i%10==0:
            print("Sampling",i)
        rand_sample = 0.3*torch.randn(1,network_params.bottleneck).float().cuda()
        rand_out = network_params.autoEncoder.decoder(rand_sample)
        if network_params.dims==2:
            z = torch.zeros(1,network_params.numPoints,1).float().cuda()
            rand_out = torch.cat((rand_out,z),2)

        if not os.path.exists(misc_variables.outputDir+"random_samples/"):
            os.makedirs(misc_variables.outputDir+"random_samples/")
        outPath = os.path.join(misc_variables.outputDir+"random_samples/",str(i)+".obj")
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(rand_out.cpu().detach().numpy()[0])
        objInstance.setFaces(meshFaces.detach().numpy())
        objInstance.saveAs(outPath)
    '''
