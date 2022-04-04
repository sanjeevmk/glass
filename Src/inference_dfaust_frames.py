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

totalRounds = 15

datalen = len(data_classes.original)

pairs = []
import random
for idx_ptr,idx in enumerate(data_classes.original.key_indices[:-1]):
    pairs.append((idx,data_classes.original.key_indices[idx_ptr+1]))
print(pairs)
minRound = training_params.test_round
if not os.path.exists(misc_variables.outputDir):
    os.makedirs(misc_variables.outputDir)
if not os.path.exists(misc_variables.randoutputDir):
    os.makedirs(misc_variables.randoutputDir)
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

    for ind,data in enumerate(data_classes.torch_original):
        pts,faces,names = data
        pts = pts[:,:,:network_params.dims]
        pts = pts.float().cuda()
        faces = faces.int().cuda()

        code = network_params.autoEncoder.encoder(pts.transpose(2,1))
        #print(names)
        #print(code)

        reconstruction = network_params.autoEncoder.decoder1(code)
        reconstruction_l = reconstruction.cpu().detach().numpy().tolist()
        reconstructions.extend(reconstruction_l)
        
        reconstructionError = losses.mse(pts,reconstruction)
        samples.extend(code)
        meshNames.extend(names)
        allPts.extend(pts)

        meshFaces = faces[0]
        meanReconError += reconstructionError.item()
        if batchIter%1==0:
            print("global:",batchIter,len(data_classes.torch_original),meanReconError/(batchIter+1))
        batchIter+=1
    meanReconError /= len(data_classes.torch_original)
    matched_frame_dists = [] 
    for index,p in enumerate(pairs):
        print(rounds,index,len(pairs))
        #if index!=14 and index!=18 and index!=24 and index!=26:
        #    continue
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
        print(sourceName,destName)
        if_dists = []
        gt_if_dists = []
        frames = []
        gt_frames = []
        codes = []
        start_vert = None
        end_vert = None
        for t in np.arange(0+1.0/1000.00,1.0+1.0/1000.0,1.0/1000.0):
            interpolatedSample = (1-t)*sourceSample + t*destSample
            interpolatedVerts = network_params.autoEncoder.decoder1(torch.unsqueeze(interpolatedSample,0))
            codes.append(interpolatedSample.cpu().detach().numpy())

            currFrame = interpolatedVerts.cpu().detach().numpy()[0]

            if t==0:
                start_vert = torch.unsqueeze(allPts[i],0) #interpolatedVerts
            if t==1:
                end_vert = torch.unsqueeze(allPts[j],0) #interpolatedVerts

            if frames:
                dist = np.sum(np.sqrt(np.sum((currFrame-frames[-1])**2,1)))
                if_dists.append(dist.item())
            else:
                if_dists.append(0)

            currFrame = interpolatedVerts.cpu().detach().numpy()[0]
            frames.append(currFrame)
        
        for frame_index in range(i,j+1):
            if gt_frames:
                dist = np.sum(np.sqrt(np.sum((allPts[frame_index].cpu().detach().numpy()-gt_frames[-1].cpu().detach().numpy())**2,1)))
                gt_if_dists.append(dist.item())
            else:
                gt_if_dists.append(0)

            gt_frames.append(allPts[frame_index])

        gt_if_dists = np.array(gt_if_dists)
        if_dists = np.array(if_dists)
        totalDist = sum(if_dists)
        gt_totalDist = sum(gt_if_dists)
        cum_dists = np.cumsum(if_dists)
        gt_cum_dists = np.cumsum(gt_if_dists)
        chosen_frames = []

        _matched_frame_dists = []
        for fnum,gt_frame in enumerate(gt_frames):
            gt_frac_dist = 1.0*gt_cum_dists[fnum]/gt_totalDist
            frame_index = np.argmin(np.abs(cum_dists/totalDist - gt_frac_dist))
            chosen_frames.append(frames[frame_index])
            _dist = np.sqrt(np.sum((frames[frame_index]-gt_frame.cpu().detach().numpy())**2,1))
            _matched_frame_dists.append(_dist)

        print("Matched frame dist:",np.mean(_matched_frame_dists))
        matched_frame_dists.append(np.mean(_matched_frame_dists))


        objInstance = Obj.Obj("dummy.obj")
        outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(0)+".obj"
        if not os.path.exists(misc_variables.outputDir+"p_"+str(index+1)+"/"):
            os.makedirs(misc_variables.outputDir+"p_"+str(index+1)+"/")
        outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),outName)
        if network_params.dims==2:
            z = torch.zeros(1,network_params.numPoints,1).float().cuda()
            start_vert = torch.cat((start_vert,z),2)
        objInstance.setVertices(torch.unsqueeze(allPts[i],0).cpu().detach().numpy()[0])
        objInstance.setFaces(meshFaces.cpu().detach().numpy())
        objInstance.saveAs(outPath)

        objInstance = Obj.Obj("dummy.obj")
        outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(len(chosen_frames)+1)+".obj"
        if not os.path.exists(misc_variables.outputDir+"p_"+str(index+1)+"/"):
            os.makedirs(misc_variables.outputDir+"p_"+str(index+1)+"/")
        outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),outName)
        if network_params.dims==2:
            z = torch.zeros(1,network_params.numPoints,1).float().cuda()
            end_vert = torch.cat((end_vert,z),2)
        objInstance.setVertices(torch.unsqueeze(allPts[j],0).cpu().detach().numpy()[0])
        objInstance.setFaces(meshFaces.cpu().detach().numpy())
        objInstance.saveAs(outPath)

        prevFrame = None
        for fnum,frame in enumerate(chosen_frames):
            objInstance = Obj.Obj("dummy.obj")
            outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(fnum+1)+".obj"
            outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),outName)
            objInstance.setVertices(frame)
            objInstance.setFaces(meshFaces.cpu().detach().numpy())
            objInstance.saveAs(outPath)
    
    
    mean_match_distance = np.mean(matched_frame_dists)
    print("Average Matched Frame Dist:",np.mean(matched_frame_dists))
    outPath = os.path.join(misc_variables.outputDir,'match_distance.txt')
    with open(outPath,'w') as f:
        f.write(str(mean_match_distance) + '\n')

    outPath = os.path.join(misc_variables.outputDir,'reconstruction.txt')
    with open(outPath,'w') as f:
        f.write(str(meanReconError) + '\n')
