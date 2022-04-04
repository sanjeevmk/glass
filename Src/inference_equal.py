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
if not os.path.exists(misc_variables.randoutputDir):
    os.makedirs(misc_variables.randoutputDir)
if not os.path.exists(misc_variables.randoutputDir[:-1]+"_hi/"):
    os.makedirs(misc_variables.randoutputDir[:-1]+"_hi/")

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
            rand_sample = torch.randn(1,network_params.bottleneck).float().cuda()
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

            #_mesh = trimesh.Trimesh(vertices=rand_out.cpu().detach().numpy()[0],faces=meshFaces.cpu().detach().numpy(),process=False)
            #mean_curv = trimesh.curvature.discrete_mean_curvature_measure(_mesh,_mesh.vertices,radius)
            #sample_smoothness_table.append(mean_curv)

            #high_rand_obj = high_res_projection.project(data_classes.template_low,data_classes.template_hi,data_classes.template_hi,rand_out.cpu().detach().numpy()[0])
            #outPath = os.path.join(misc_variables.randoutputDir[:-1]+ "_hi/",str(i)+".obj")
            #high_rand_obj.saveAs(outPath)


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

                #_mesh = trimesh.Trimesh(vertices=normed_vertices,faces=meshFaces.cpu().detach().numpy(),process=False)
                #mean_curv = trimesh.curvature.discrete_mean_curvature_measure(_mesh,_mesh.vertices,radius)
                #sample_smoothness_table.append(mean_curv)

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

    interp_smoothness_table = []
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

        #src_high_res_path = [_name for _name in data_classes.high_prefix if _name in sourceName]
        #src_high_res_path = src_high_res_path[0]
        #src_high_res_path = os.path.join(data_classes.high_folder,src_high_res_path+".obj")

        #dest_high_res_path = [_name for _name in data_classes.high_prefix if _name in destName]
        #dest_high_res_path = dest_high_res_path[0]
        #dest_high_res_path = os.path.join(data_classes.high_folder,dest_high_res_path+".obj")

        if_dists = []
        frames = []
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
                #high_start_obj = high_res_projection.project(data_classes.template_low,data_classes.template_hi,src_high_res_path,start_vert.cpu().detach().numpy()[0])
                #high_start_vert =  high_start_obj.vertices
            if t==1:
                end_vert = torch.unsqueeze(allPts[j],0) #interpolatedVerts
                #high_end_obj = high_res_projection.project(data_classes.template_low,data_classes.template_hi,dest_high_res_path,end_vert.cpu().detach().numpy()[0])
                #high_end_vert =  high_end_obj.vertices

            if frames:
                dist = np.sum(np.sqrt(np.sum((currFrame-frames[-1])**2,1)))
                if_dists.append(dist.item())
            else:
                if_dists.append(0)

            currFrame = interpolatedVerts.cpu().detach().numpy()[0]
            frames.append(currFrame)


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

            #chosen_frames.append(frames[frame_index])
            chosen_frames.append(frames[frame_index])

            #_mesh = trimesh.Trimesh(vertices=frames[frame_index],faces=meshFaces.cpu().detach().numpy(),process=False)
            #mean_curv = trimesh.curvature.discrete_mean_curvature_measure(_mesh,frames[frame_index],radius)
            #curvatures.append(mean_curv)

            '''
            if fnum < num_frames/2:
                high_frame_obj = high_res_projection.project(data_classes.template_low,data_classes.template_hi,src_high_res_path,frames[frame_index])
                high_frames.append(high_frame_obj.vertices)
            else:
                high_frame_obj = high_res_projection.project(data_classes.template_low,data_classes.template_hi,dest_high_res_path,frames[frame_index])
                high_frames.append(high_frame_obj.vertices)
            '''

        #print(np.mean(curvatures))
        interp_smoothness_table.append(np.mean(curvatures))
        if not os.path.exists(misc_variables.outputDir+"p_"+str(index+1)+"/"):
            os.makedirs(misc_variables.outputDir+"p_"+str(index+1)+"/")

        objInstance = Obj.Obj("dummy.obj")
        outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(0)+".obj"
        outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),outName)
        if network_params.dims==2:
            z = torch.zeros(1,network_params.numPoints,1).float().cuda()
            start_vert = torch.cat((start_vert,z),2)
        objInstance.setVertices(torch.unsqueeze(allPts[i],0).cpu().detach().numpy()[0])
        objInstance.setFaces(meshFaces.cpu().detach().numpy())
        objInstance.saveAs(outPath)

        objInstance = Obj.Obj("dummy.obj")
        outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(num_frames-1)+".obj"
        outPath = os.path.join(misc_variables.outputDir+"p_"+str(index+1),outName)
        if network_params.dims==2:
            z = torch.zeros(1,network_params.numPoints,1).float().cuda()
            end_vert = torch.cat((end_vert,z),2)
        objInstance.setVertices(torch.unsqueeze(allPts[j],0).cpu().detach().numpy()[0])
        #objInstance.setVertices(end_vert.cpu().detach().numpy()[0])
        objInstance.setFaces(meshFaces.cpu().detach().numpy())
        objInstance.saveAs(outPath)

        '''
        objInstance = Obj.Obj("dummy.obj")
        outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(0)+".obj"
        if not os.path.exists(misc_variables.outputDir[:-1]+"_hi/"+"p_"+str(index+1)+"/"):
            os.makedirs(misc_variables.outputDir[:-1]+"_hi/"+"p_"+str(index+1)+"/")
        outPath = os.path.join(misc_variables.outputDir[:-1]+"_hi/"+"p_"+str(index+1),outName)
        objInstance.setVertices(high_start_vert)
        objInstance.setFaces(data_classes.high_faces)
        objInstance.saveAs(outPath)

        objInstance = Obj.Obj("dummy.obj")
        outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(num_frames-1)+".obj"
        if not os.path.exists(misc_variables.outputDir[:-1]+"_hi/"+"p_"+str(index+1)+"/"):
            os.makedirs(misc_variables.outputDir[:-1]+"_hi/"+"p_"+str(index+1)+"/")
        outPath = os.path.join(misc_variables.outputDir[:-1]+"_hi/"+"p_"+str(index+1),outName)
        objInstance.setVertices(high_end_vert)
        objInstance.setFaces(data_classes.high_faces)
        objInstance.saveAs(outPath)
        '''

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
        '''
        for fnum,frame in enumerate(high_frames):
            objInstance = Obj.Obj("dummy.obj")
            outName = "r"+str(rounds)+"_"+sourceName + "_" + destName + "_" + str(fnum+1)+".obj"
            outPath = os.path.join(misc_variables.outputDir[:-1]+"_hi/"+"p_"+str(index+1),outName)
            objInstance.setVertices(frame)
            objInstance.setFaces(data_classes.high_faces)
            objInstance.saveAs(outPath)

        outPath = os.path.join(misc_variables.outputDir[:-1]+"_hi/"+"p_"+str(index+1),"arap.txt")
        csv.writer(open(outPath,'w'),delimiter='\n').writerows(arapTable)
        outPath = os.path.join(misc_variables.outputDir[:-1]+"_hi/"+"p_"+str(index+1),"dist.txt")
        csv.writer(open(outPath,'w'),delimiter='\n').writerows(distanceTable)
        '''
    outPath = os.path.join(misc_variables.outputDir,"sample_smoothness.txt")
    if len(sample_smoothness_table)>0:
        with open(outPath,'w') as f:
            f.write(str(np.mean(sample_smoothness_table))+'\n')

    outPath = os.path.join(misc_variables.outputDir,"sample_dist.txt")
    if len(sample_coverage_table)>0:
        with open(outPath,'w') as f:
            f.write(str(np.mean(sample_coverage_table))+'\n')

    outPath = os.path.join(misc_variables.outputDir,"interp_smoothness.txt")
    if len(interp_smoothness_table)>0:
        with open(outPath,'w') as f:
            f.write(str(np.mean(interp_smoothness_table))+'\n')

    outPath = os.path.join(misc_variables.outputDir,"interp_dist.txt")
    if len(interp_dist_table)>0:
        with open(outPath,'w') as f:
            f.write(str(np.mean(interp_dist_table))+'\n')

    outPath = os.path.join(misc_variables.outputDir,"interp_arap.txt")
    if len(interp_arap_table)>0:
        with open(outPath,'w') as f:
            f.write(str(np.mean(interp_arap_table))+'\n')

