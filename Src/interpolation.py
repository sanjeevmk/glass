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

test_pairs = []
if data_classes.testset_original is not None:
    datalen = len(data_classes.testset_original)
    test_pairs = []
    import random
    for i in range(datalen):
        for j in range(datalen):
            if i==j:
                continue
            if (j,i) in test_pairs:
                continue
            options = [x for x in range(datalen) if x!=i and x!=j]
            #third = random.choice(options)
            test_pairs.append((i,j)) #,third))

#pairs = [pairs[41],pairs[42]]
minRound = training_params.test_round
if not os.path.exists(misc_variables.outputDir):
    os.makedirs(misc_variables.outputDir)
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

    interp_dist_table = []
    interp_arap_table = []

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

        if_dists = []
        frames = []
        ext_frames_pre = []
        ext_frames_post = []
        codes = []
        start_vert = None
        high_start_vert = None
        end_vert = None
        high_end_vert = None

        for t in np.arange(1.0/1000.0,1.0+(1.0/1000.0),1.0/1000.0):
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

        for t in np.arange(-2.0,0.0,2.0/30.0):
            interpolatedSample = (1-t)*sourceSample + t*destSample
            interpolatedVerts = network_params.autoEncoder.decoder1(torch.unsqueeze(interpolatedSample,0))

            currFrame = interpolatedVerts.cpu().detach().numpy()[0]
            ext_frames_pre.append(currFrame)

        for t in np.arange(1.0+2.0/30.0,3.0,2.0/30.0):
            interpolatedSample = (1-t)*sourceSample + t*destSample
            interpolatedVerts = network_params.autoEncoder.decoder1(torch.unsqueeze(interpolatedSample,0))

            currFrame = interpolatedVerts.cpu().detach().numpy()[0]
            ext_frames_post.append(currFrame)

        num_frames = 32
        if_dists = np.array(if_dists)
        totalDist = sum(if_dists)
        cum_dists = np.cumsum(if_dists)
        chosen_frames = []
        #high_frames = []
        curvatures = []
        frame_index=0
        for fnum in range(1,num_frames-1):
            print(fnum)
            reqd_dist = fnum*float(totalDist/num_frames)
            frame_index = np.argmin(np.abs(cum_dists - reqd_dist))
            dist = cum_dists[frame_index]

            chosen_frames.append(frames[frame_index])

        #print(np.mean(curvatures))
        if not os.path.exists(misc_variables.outputDir+"p_"+str(index+1)+"/"):
            os.makedirs(misc_variables.outputDir+"p_"+str(index+1)+"/")

        objInstance = Obj.Obj("dummy.obj")
        outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(0)+".obj"
        outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),outName)
        objInstance.setVertices(torch.unsqueeze(allPts[i],0).cpu().detach().numpy()[0])
        objInstance.setFaces(meshFaces.cpu().detach().numpy())
        objInstance.saveAs(outPath)

        objInstance = Obj.Obj("dummy.obj")
        outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(num_frames-1)+".obj"
        outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),outName)
        objInstance.setVertices(torch.unsqueeze(allPts[j],0).cpu().detach().numpy()[0])
        objInstance.setFaces(meshFaces.cpu().detach().numpy())
        objInstance.saveAs(outPath)

        arapTable = []
        distanceTable = []
        prevFrame = None
        for fnum,frame in enumerate(chosen_frames):
            objInstance = Obj.Obj("dummy.obj")
            outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(fnum+1)+".obj"
            outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),outName)
            objInstance.setVertices(frame)
            objInstance.setFaces(meshFaces.cpu().detach().numpy())
            objInstance.saveAs(outPath)

            torch_frame = torch.unsqueeze(torch.from_numpy(frame).float().cuda(),0)
            i = p[0]
            j = p[1]
            energyLoss1,_ = losses.arap(torch.unsqueeze(allPts[i],0),torch_frame,torch.unsqueeze(allneighborsMatrix[i],0),torch.unsqueeze(allnumNeighbors[i],0),torch.unsqueeze(allaccnumNeighbors[i],0),torch.unsqueeze(allweightMatrix[i],0),network_params.testEnergyWeight)
            energyLoss2,_ = losses.arap(torch.unsqueeze(allPts[j],0),torch_frame,torch.unsqueeze(allneighborsMatrix[j],0),torch.unsqueeze(allnumNeighbors[j],0),torch.unsqueeze(allaccnumNeighbors[j],0),torch.unsqueeze(allweightMatrix[j],0),network_params.testEnergyWeight)
            energy = 0.5*(energyLoss1.mean() + energyLoss2.mean()).item()

            if prevFrame is not None:
                dist = np.sum(np.sqrt(np.sum((frame-prevFrame)**2,1)))
                distanceTable.append([dist])
            arapTable.append([energy])
            prevFrame = frame

        interp_dist_table.append(np.std(np.array(distanceTable).squeeze()))
        interp_arap_table.append(np.mean(np.array(arapTable).squeeze()))
        outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),"arap.txt")
        csv.writer(open(outPath,'w'),delimiter='\n').writerows(arapTable)
        outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),"dist.txt")
        csv.writer(open(outPath,'w'),delimiter='\n').writerows(distanceTable)

        if not os.path.exists(extrapolation_dir+"p_"+str(index+1)+"/"):
            os.makedirs(extrapolation_dir+"p_"+str(index+1)+"/")

        for fnum,frame in enumerate(ext_frames_pre):
            objInstance = Obj.Obj("dummy.obj")
            outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(fnum+1)+"_pre.obj"
            outPath = os.path.join(extrapolation_dir+"p_"+str(index+1),outName)
            objInstance.setVertices(frame)
            objInstance.setFaces(meshFaces.cpu().detach().numpy())
            objInstance.saveAs(outPath)

        for fnum,frame in enumerate(ext_frames_post):
            objInstance = Obj.Obj("dummy.obj")
            outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(fnum+1)+"_post.obj"
            outPath = os.path.join(extrapolation_dir+"p_"+str(index+1),outName)
            objInstance.setVertices(frame)
            objInstance.setFaces(meshFaces.cpu().detach().numpy())
            objInstance.saveAs(outPath)

    outPath = os.path.join(misc_variables.outputDir,"interp_dist.txt")
    if len(interp_dist_table)>0:
        with open(outPath,'w') as f:
            f.write(str(np.mean(interp_dist_table))+'\n')

    outPath = os.path.join(misc_variables.outputDir,"interp_arap.txt")
    if len(interp_arap_table)>0:
        with open(outPath,'w') as f:
            f.write(str(np.mean(interp_arap_table))+'\n')
    if True: #not test_pairs:
        print("No Test Set, exit")
        exit()

    test_outputDir = misc_variables.outputDir.replace("interpolation","interpolation_test")
    #TEST SET
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

    for ind,data in enumerate(data_classes.testset_torch_full_arap_original):
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
            print("global:",batchIter,len(data_classes.testset_torch_full_arap_original),meanReconError/(batchIter+1))
        batchIter+=1

    interp_dist_table = []
    interp_arap_table = []

    for index,p in enumerate(test_pairs):
        print(rounds,index,len(test_pairs))
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

        if_dists = []
        frames = []
        ext_frames_pre = []
        ext_frames_post = []
        codes = []
        start_vert = None
        high_start_vert = None
        end_vert = None
        high_end_vert = None

        for t in np.arange(1.0/1000.0,1.0+(1.0/1000.0),1.0/1000.0):
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

        for t in np.arange(-1.0,0.0,1.0/30.0):
            interpolatedSample = (1-t)*sourceSample + t*destSample
            interpolatedVerts = network_params.autoEncoder.decoder1(torch.unsqueeze(interpolatedSample,0))

            currFrame = interpolatedVerts.cpu().detach().numpy()[0]
            ext_frames_pre.append(currFrame)

        for t in np.arange(1.0+1.0/30.0,2.0,1.0/30.0):
            interpolatedSample = (1-t)*sourceSample + t*destSample
            interpolatedVerts = network_params.autoEncoder.decoder1(torch.unsqueeze(interpolatedSample,0))

            currFrame = interpolatedVerts.cpu().detach().numpy()[0]
            ext_frames_post.append(currFrame)

        num_frames = 32
        if_dists = np.array(if_dists)
        totalDist = sum(if_dists)
        cum_dists = np.cumsum(if_dists)
        chosen_frames = []
        #high_frames = []
        curvatures = []
        frame_index=0
        for fnum in range(1,num_frames-1):
            print(fnum)
            reqd_dist = fnum*float(totalDist/num_frames)
            frame_index = np.argmin(np.abs(cum_dists - reqd_dist))
            dist = cum_dists[frame_index]

            chosen_frames.append(frames[frame_index])

        if not os.path.exists(test_outputDir+"p_"+str(index+1)+"/"):
            os.makedirs(test_outputDir+"p_"+str(index+1)+"/")

        objInstance = Obj.Obj("dummy.obj")
        outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(0)+".obj"
        outPath = os.path.join(test_outputDir+"p_"+str(index+1),outName)
        objInstance.setVertices(torch.unsqueeze(allPts[i],0).cpu().detach().numpy()[0])
        objInstance.setFaces(meshFaces.cpu().detach().numpy())
        objInstance.saveAs(outPath)

        objInstance = Obj.Obj("dummy.obj")
        outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(num_frames-1)+".obj"
        outPath = os.path.join(test_outputDir+"p_"+str(index+1),outName)
        objInstance.setVertices(torch.unsqueeze(allPts[j],0).cpu().detach().numpy()[0])
        objInstance.setFaces(meshFaces.cpu().detach().numpy())
        objInstance.saveAs(outPath)

        arapTable = []
        distanceTable = []
        prevFrame = None
        for fnum,frame in enumerate(chosen_frames):
            objInstance = Obj.Obj("dummy.obj")
            outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(fnum+1)+".obj"
            outPath = os.path.join(test_outputDir+"p_"+str(index+1),outName)
            objInstance.setVertices(frame)
            objInstance.setFaces(meshFaces.cpu().detach().numpy())
            objInstance.saveAs(outPath)

            torch_frame = torch.unsqueeze(torch.from_numpy(frame).float().cuda(),0)
            i = p[0]
            j = p[1]
            energyLoss1,_ = losses.arap(torch.unsqueeze(allPts[i],0),torch_frame,torch.unsqueeze(allneighborsMatrix[i],0),torch.unsqueeze(allnumNeighbors[i],0),torch.unsqueeze(allaccnumNeighbors[i],0),torch.unsqueeze(allweightMatrix[i],0),network_params.testEnergyWeight)
            energyLoss2,_ = losses.arap(torch.unsqueeze(allPts[j],0),torch_frame,torch.unsqueeze(allneighborsMatrix[j],0),torch.unsqueeze(allnumNeighbors[j],0),torch.unsqueeze(allaccnumNeighbors[j],0),torch.unsqueeze(allweightMatrix[j],0),network_params.testEnergyWeight)
            energy = 0.5*(energyLoss1.mean() + energyLoss2.mean()).item()

            if prevFrame is not None:
                dist = np.sum(np.sqrt(np.sum((frame-prevFrame)**2,1)))
                distanceTable.append([dist])
            arapTable.append([energy])
            prevFrame = frame

        interp_dist_table.append(np.std(np.array(distanceTable).squeeze()))
        interp_arap_table.append(np.mean(np.array(arapTable).squeeze()))
        outPath = os.path.join(test_outputDir+"p_"+str(index+1),"arap.txt")
        csv.writer(open(outPath,'w'),delimiter='\n').writerows(arapTable)
        outPath = os.path.join(test_outputDir+"p_"+str(index+1),"dist.txt")
        csv.writer(open(outPath,'w'),delimiter='\n').writerows(distanceTable)

        if not os.path.exists(extrapolation_dir_test+"p_"+str(index+1)+"/"):
            os.makedirs(extrapolation_dir_test+"p_"+str(index+1)+"/")

        for fnum,frame in enumerate(ext_frames_pre):
            objInstance = Obj.Obj("dummy.obj")
            outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(fnum+1)+"_pre.obj"
            outPath = os.path.join(extrapolation_dir_test+"p_"+str(index+1),outName)
            objInstance.setVertices(frame)
            objInstance.setFaces(meshFaces.cpu().detach().numpy())
            objInstance.saveAs(outPath)

        for fnum,frame in enumerate(ext_frames_post):
            objInstance = Obj.Obj("dummy.obj")
            outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(fnum+1)+"_post.obj"
            outPath = os.path.join(extrapolation_dir_test+"p_"+str(index+1),outName)
            objInstance.setVertices(frame)
            objInstance.setFaces(meshFaces.cpu().detach().numpy())
            objInstance.saveAs(outPath)

    outPath = os.path.join(test_outputDir,"interp_dist.txt")
    if len(interp_dist_table)>0:
        with open(outPath,'w') as f:
            f.write(str(np.mean(interp_dist_table))+'\n')

    outPath = os.path.join(test_outputDir,"interp_arap.txt")
    if len(interp_arap_table)>0:
        with open(outPath,'w') as f:
            f.write(str(np.mean(interp_arap_table))+'\n')

