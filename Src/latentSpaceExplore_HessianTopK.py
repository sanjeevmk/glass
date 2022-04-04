import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs

import Energies
import Leapfrog
#import Hessian 
import Hessian_bottomk as Hessian
import Hessian_topk as Hessian_topk
from Utils import writer,Off,Obj,Pts,rect_remesh
import header 
from torch.autograd import Variable
import math

from RegArap import ARAP as ArapReg

def hessianExplore(training_params,data_classes,network_params,losses,misc_variables,rounds):
    for parameter in network_params.autoEncoder.decoder1.parameters():
        parameter.requires_grad = False
    shapes = []
    meshNames = []
    for ind,data in enumerate(data_classes.expanded):
        #pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,area,names,_,_ = data
        pts,faces,names = data
        pts = torch.unsqueeze(torch.from_numpy(pts).float().cuda(),0)
        pts = pts[:,:,:3]
        meshNames.extend(names)
        shapes.extend(pts)

    _,template_face,_ = data_classes.original[0]
    arapreg = ArapReg(template_face,network_params.numPoints).cuda()
    names = meshNames[0]

    print("Round:",rounds," HMC",flush=True)
    network_params.autoEncoder.eval()
    network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))

    if not os.path.exists(misc_variables.sampleDir):
        os.makedirs(misc_variables.sampleDir)

    def dumpSamples(outDir,samples,p_pts=None,corner=-1):
        for i in range(len(samples)):
            objInstance = Obj.Obj("dummy.obj")
            '''
            newPoints,newFaces,corner_vid = rect_remesh.rectRemesh(samples[i],faces,corner_vid=corner)
            objInstance.setCorner(corner_vid)
            '''
            objInstance.setVertices(samples[i])
            objInstance.setFaces(faces)
            objInstance.saveAs(os.path.join(outDir,str(rounds) + "_" + names.split(".")[0] + "_" + str(i)+".obj"))

            if p_pts is not None:
                ptsInstance = Pts.Pts("dummy.pts")
                ptsInstance.setVertices(p_pts)
                ptsInstance.saveAs(os.path.join(outDir,str(rounds) + "_" + names.split(".")[0] + "_" + str(i)+".pts"))

    def dumpSamples_top(outDir,samples,p_pts=None,corner=-1):
        for i in range(len(samples)):
            objInstance = Obj.Obj("dummy.obj")
            '''
            newPoints,newFaces,corner_vid = rect_remesh.rectRemesh(samples[i],faces,corner_vid=corner)
            objInstance.setCorner(corner_vid)
            '''
            objInstance.setVertices(samples[i])
            objInstance.setFaces(faces)
            objInstance.saveAs(os.path.join(outDir,str(rounds) + "_" + names.split(".")[0] + "_" + str(i)+"_topk.obj"))

    for ind,data in enumerate(data_classes.full_arap):
        hessian_topk = Hessian_topk.Hessian(training_params.numsteps,0.1)

        network_params.autoEncoder.eval()
        samples = []
        energies = []
        print("HMC Source Data:",ind,len(data_classes.expanded),flush=True)
        pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,names,square_weightMatrix = data
        print(pts.shape,names)
        query = np.expand_dims(pts[:,:3],0)
        query = np.reshape(query,(1,-1))
        pts = torch.unsqueeze(torch.from_numpy(pts).float().cuda(),0)
        pts = pts[:,:,:3]

        numNeighbors = torch.unsqueeze(torch.from_numpy(numNeighbors).int().cuda(),0)
        accnumNeighbors = torch.unsqueeze(torch.from_numpy(accnumNeighbors).int().cuda(),0)
        neighborsMatrix = torch.unsqueeze(torch.from_numpy(neighborsMatrix).int().cuda(),0)
        weightMatrix = torch.unsqueeze(torch.from_numpy(weightMatrix).float().cuda(),0)
        square_weightMatrix = -1.0*torch.from_numpy(square_weightMatrix).float().cuda()

        network_params.autoEncoder.zero_grad()

        energy_fn = None
        if training_params.energy=='arap':
            #energy_fn = Energies.ArapEnergy(pts,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
            energy_fn = Energies.ArapEnergyHessian(pts[:,:,:3],neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,template_face,network_params.numPoints,network_params.bottleneck,network_params.testEnergyWeight)
        if training_params.energy=='arap_anal':
            energy_fn = Energies.ArapEnergyHessianAnalytical(pts[:,:,:3],neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,template_face,network_params.numPoints,network_params.bottleneck,network_params.testEnergyWeight)
        if training_params.energy=='regarap':
            energy_fn = Energies.ArapRegHessian(template_face,network_params.numPoints,nz_max=network_params.bottleneck)
        if training_params.energy=='iso':
            energy_fn = Energies.IsometricEnergy(pts,neighborsMatrix,numNeighbors,network_params.testEnergyWeight)
        if training_params.energy=='arap2d':
            energy_fn = Energies.ArapEnergy2D(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='asap':
            energy_fn = Energies.AsapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='asap2d':
            energy_fn = Energies.AsapEnergy2D(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        #seedCode = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
        seedCode = network_params.autoEncoder.encoder(pts.transpose(2,1))

        recon = network_params.autoEncoder.decoder1(seedCode)
        reconstructionError = losses.mse(pts,recon)
        print("Reconstruction Error:",reconstructionError)

        #recon = network_params.autoEncoder.decoder2(pts,recon,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(recon.cpu().detach().numpy()[0])
        objInstance.setFaces(faces)
        objInstance.saveAs(os.path.join(misc_variables.reconDir,str(rounds) + "_" + names.split(".")[0] + "_recon.obj"))

        stepSize = training_params.stepsize
        stepsizeMin = 0.001
        stepsizeMax = 0.1
        stepsizeInc = 1.02
        stepsizeDec = 0.98
        targetAcceptanceRate = 0.7
        avgAcceptanceSlowness = 0.9
        avgAcceptance = 0.0
        burnin = 0

        code = seedCode.clone().detach()

        from datetime import datetime
        import time
        start = datetime.now()
        for i in range(19): #training_params.hmcEpochs):
            newCode = hessian_topk(code,energy_fn,stepSize=stepSize,k=training_params.ncomp,decoder=network_params.autoEncoder)
            if isinstance(newCode,str):
                if newCode == "eig failed":
                    return "eig failed"
            #newCode,energy = hessian(code,energy_fn,stepSize=stepSize,k=training_params.ncomp,decoder=network_params.autoEncoder)

            newSample = network_params.autoEncoder.decoder1(newCode)
            #newSample = network_params.autoEncoder.decoder2(pts[:,:,:3],newSample,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
            network_params.autoEncoder.zero_grad()
            code = seedCode.clone().detach()
            if network_params.dims == 2:
                z = torch.zeros(1,network_params.numPoints,1).float().cuda()
                newSample = torch.cat((newSample,z),2)

            if np.isnan(newSample.cpu().detach().numpy()[0]).any():
                continue
            #energies.append(energy/network_params.testEnergyWeight.item())
            samples.append(newSample.cpu().detach().numpy()[0])

        dumpSamples_top(misc_variables.hmcsampleDir,samples)
        end = datetime.now()
        print("HMC Time:",(end-start).seconds)
        #if len(energies) == 0:
        #    continue
        #print("min energy:",min(energies))

        #if math.isnan(energies[0].item()):
        #    continue

        if len(samples)==0:
            continue

        start = datetime.now()

        selectedSamples = np.array([s.cpu().detach().numpy() for i,s in enumerate(shapes) if meshNames[i]!=names])
        selectedSamples = np.reshape(selectedSamples,(selectedSamples.shape[0],-1))
        initialLen = len(selectedSamples)
        samples = np.array(samples)
        mmr_lambda=0.1
        print("Round:",rounds,len(samples)," SAMPLE SELECTION")
        if len(samples)>0:
            for numAddedSamples in range(misc_variables.numNewSamples):
                print("sample adding:",numAddedSamples,misc_variables.numNewSamples)
                allDistances = []
                for s in samples:
                    flatS = np.reshape(s[:,:3],(1,-1))
                    QDist = np.mean((query- flatS)**2,1)[0]
                    SSimilarity = np.max(cs(selectedSamples,flatS))
                    QSimilarity = np.max(cs(query,flatS))
                    similarity = mmr_lambda*QSimilarity - (1-mmr_lambda)*SSimilarity
                    allDistances.append(similarity)

                allDistances = np.array(allDistances)
                index = np.argmax(allDistances)
                #print(index,len(energies),allDistances.shape)
                print(index,allDistances.shape)
                selectedS = samples[index,:,:]
                #print("selected energy:",energies[index])
                samples = np.delete(samples,index,axis=0)
                selectedSamples = np.vstack((selectedSamples,np.reshape(selectedS[:,:3],(1,-1))))

        selectedSamples = selectedSamples[initialLen:,:]
        selectedSamples = np.reshape(selectedSamples,(selectedSamples.shape[0],network_params.numPoints,3))
        end = datetime.now()
        print("MMR Time:",(end-start).seconds)

        if network_params.dims==2:
            z = np.zeros((selectedSamples.shape[0],network_params.numPoints,1))
            selectedSamples = np.concatenate((selectedSamples,z),2)

        if training_params.project:
            #dumpSamples(misc_variables.hmcsampleDir,selectedSamples)

            bestShape = None
            perturbedShape = np.array(selectedSamples)[:,:,:3]
            perturbedShape = torch.from_numpy(perturbedShape).float().cuda()

            bestEnergy = 1e9
            prevEnergy = 1e9
            for i in range(training_params.optEpochs):
                energy = None
                if training_params.energy == 'arap' or training_params.energy == 'regarap' or training_params.energy == 'arap_anal':
                    energy,rotations = losses.arap(pts[:,:,:3],perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
                    #updated_rotations = losses.bending(perturbedShape,pts[:,:,:3],neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,rotations,network_params.testEnergyWeight)
                    rhs = losses.arap_solve_rhs(pts[:,:,:3],perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,rotations,network_params.testEnergyWeight)
                    #perturbedShape = torch.unsqueeze(torch.mm(torch.inverse(square_weightMatrix),rhs),0)
                    perturbedShape = torch.linalg.solve(square_weightMatrix,rhs)
                    perturbedShape = torch.unsqueeze(perturbedShape,0)
                    energy,_ = losses.arap(perturbedShape,pts[:,:,:3],neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
                    energy = energy.mean()

                if energy.item() < bestEnergy:
                    bestEnergy = energy.item()/network_params.testEnergyWeight.item()
                    bestShape = perturbedShape 

                if abs(bestEnergy) < training_params.bestenergy:
                    break
                
                prevEnergy = energy.item()
            bestShape_closed = bestShape.clone().detach()
            bestEnergy_closed = bestEnergy

            print("perturbed",i,bestEnergy,flush=True)
            if bestShape is None:
                continue

            if bestEnergy>training_params.bestenergy:
                bestEnergy = 1e9
                bestShape = None
                perturbedShape = np.array(selectedSamples)[:,:,:3]
                perturbedShape = torch.from_numpy(perturbedShape).float().cuda()
                proposedShape = header.Position(perturbedShape.view(1,-1))
                for parameter in network_params.autoEncoder.encoder.parameters():
                    parameter.requires_grad = False
                codeOptimizer = torch.optim.Adam(proposedShape.parameters(),lr=1e-3,weight_decay=0 )

                for i in range(training_params.optEpochs):
                    codeOptimizer.zero_grad()

                    energy = None
                    if training_params.energy == 'arap' or training_params.energy == 'arap_anal':
                        energy,_ = losses.arap(pts[:,:,:3],proposedShape.proposed.view(1,network_params.numPoints,3),neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
                        energy = energy.mean()

                    energy.backward()
                    codeOptimizer.step()
                    if energy.item() < bestEnergy:
                        bestEnergy = energy.item()/network_params.testEnergyWeight.item()
                        bestShape = proposedShape.proposed.view(1,network_params.numPoints,3)

                    if bestEnergy < training_params.bestenergy:
                        break

                for parameter in network_params.autoEncoder.encoder.parameters():
                    parameter.requires_grad = True
                print("perturbed optimizer",i,bestEnergy,flush=True)
            bestShape_opt = bestShape.clone().detach()
            bestEnergy_opt = bestEnergy

            bestShape = bestShape_opt
            if bestEnergy_opt>1e-3: # and bestEnergy_opt>1e-3:
                continue
            if bestEnergy_closed < bestEnergy_opt:
                bestShape = bestShape_closed
            else:
                bestShape = bestShape_opt

            selectedSamples = bestShape.cpu().detach().numpy()

            if network_params.dims==2:
                z = np.zeros((selectedSamples.shape[0],network_params.numPoints,1))
                selectedSamples = np.concatenate((selectedSamples,z),2)
            dumpSamples(misc_variables.sampleDir,selectedSamples)
        else:
            dumpSamples(misc_variables.sampleDir,selectedSamples)
    exit()
    for ind,data in enumerate(data_classes.full_arap):
        hessian_topk = Hessian_topk.Hessian(training_params.numsteps,0.1)
        network_params.autoEncoder.eval()
        samples = []
        energies = []
        print("HMC Source Data:",ind,len(data_classes.expanded),flush=True)
        pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,names,square_weightMatrix = data
        print(pts.shape,names)
        query = np.expand_dims(pts[:,:3],0)
        query = np.reshape(query,(1,-1))
        pts = torch.unsqueeze(torch.from_numpy(pts).float().cuda(),0)
        pts = pts[:,:,:3]

        numNeighbors = torch.unsqueeze(torch.from_numpy(numNeighbors).int().cuda(),0)
        accnumNeighbors = torch.unsqueeze(torch.from_numpy(accnumNeighbors).int().cuda(),0)
        neighborsMatrix = torch.unsqueeze(torch.from_numpy(neighborsMatrix).int().cuda(),0)
        weightMatrix = torch.unsqueeze(torch.from_numpy(weightMatrix).float().cuda(),0)
        square_weightMatrix = -1.0*torch.from_numpy(square_weightMatrix).float().cuda()

        network_params.autoEncoder.zero_grad()

        energy_fn = None
        if training_params.energy=='arap':
            #energy_fn = Energies.ArapEnergy(pts,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
            energy_fn = Energies.ArapEnergyHessian(pts[:,:,:3],neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,template_face,network_params.numPoints,network_params.bottleneck,network_params.testEnergyWeight)
        if training_params.energy=='arap_anal':
            energy_fn = Energies.ArapEnergyHessianAnalytical(pts[:,:,:3],neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,template_face,network_params.numPoints,network_params.bottleneck,network_params.testEnergyWeight)
        if training_params.energy=='regarap':
            energy_fn = Energies.ArapRegHessian(template_face,network_params.numPoints,nz_max=network_params.bottleneck)
        if training_params.energy=='iso':
            energy_fn = Energies.IsometricEnergy(pts,neighborsMatrix,numNeighbors,network_params.testEnergyWeight)
        if training_params.energy=='arap2d':
            energy_fn = Energies.ArapEnergy2D(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='asap':
            energy_fn = Energies.AsapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='asap2d':
            energy_fn = Energies.AsapEnergy2D(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        #seedCode = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
        seedCode = network_params.autoEncoder.encoder(pts.transpose(2,1))

        recon = network_params.autoEncoder.decoder1(seedCode)
        reconstructionError = losses.mse(pts,recon)
        print("Reconstruction Error:",reconstructionError)

        #recon = network_params.autoEncoder.decoder2(pts,recon,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(recon.cpu().detach().numpy()[0])
        objInstance.setFaces(faces)
        objInstance.saveAs(os.path.join(misc_variables.reconDir,str(rounds) + "_" + names.split(".")[0] + "_recon.obj"))

        stepSize = training_params.stepsize
        stepsizeMin = 0.001
        stepsizeMax = 0.1
        stepsizeInc = 1.02
        stepsizeDec = 0.98
        targetAcceptanceRate = 0.7
        avgAcceptanceSlowness = 0.9
        avgAcceptance = 0.0
        burnin = 0

        code = seedCode.clone().detach()

        from datetime import datetime
        import time
        start = datetime.now()
        for i in range(9): #training_params.hmcEpochs):
            newCode = hessian_topk(code,energy_fn,stepSize=stepSize,k=training_params.ncomp,decoder=network_params.autoEncoder)
            if isinstance(newCode,str):
                if newCode == "eig failed":
                    return "eig failed"

            newSample = network_params.autoEncoder.decoder1(newCode)
            #newSample = network_params.autoEncoder.decoder2(pts[:,:,:3],newSample,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
            network_params.autoEncoder.zero_grad()
            code = seedCode.clone().detach()
            if network_params.dims == 2:
                z = torch.zeros(1,network_params.numPoints,1).float().cuda()
                newSample = torch.cat((newSample,z),2)

            if np.isnan(newSample.cpu().detach().numpy()[0]).any():
                continue
            #energies.append(energy/network_params.testEnergyWeight.item())
            samples.append(newSample.cpu().detach().numpy()[0])

        dumpSamples_top(misc_variables.hmcsampleDir,samples)
        end = datetime.now()
        print("HMC Time:",(end-start).seconds)
        #if len(energies) == 0:
        #    continue
        #print("min energy:",min(energies))

        #if math.isnan(energies[0].item()):
        #    continue

        if len(samples)==0:
            continue

        start = datetime.now()

        selectedSamples = np.array([s.cpu().detach().numpy() for i,s in enumerate(shapes) if meshNames[i]!=names])
        selectedSamples = np.reshape(selectedSamples,(selectedSamples.shape[0],-1))
        initialLen = len(selectedSamples)
        samples = np.array(samples)
        mmr_lambda=0.1
        print("Round:",rounds,len(samples)," SAMPLE SELECTION")
        if len(samples)>0:
            for numAddedSamples in range(misc_variables.numNewSamples):
                print("sample adding:",numAddedSamples,misc_variables.numNewSamples)
                allDistances = []
                for s in samples:
                    flatS = np.reshape(s[:,:3],(1,-1))
                    QDist = np.mean((query- flatS)**2,1)[0]
                    SSimilarity = np.max(cs(selectedSamples,flatS))
                    QSimilarity = np.max(cs(query,flatS))
                    similarity = mmr_lambda*QSimilarity - (1-mmr_lambda)*SSimilarity
                    allDistances.append(similarity)

                allDistances = np.array(allDistances)
                index = np.argmax(allDistances)
                #print(index,len(energies),allDistances.shape)
                print(index,allDistances.shape)
                selectedS = samples[index,:,:]
                #print("selected energy:",energies[index])
                samples = np.delete(samples,index,axis=0)
                selectedSamples = np.vstack((selectedSamples,np.reshape(selectedS[:,:3],(1,-1))))

        selectedSamples = selectedSamples[initialLen:,:]
        selectedSamples = np.reshape(selectedSamples,(selectedSamples.shape[0],network_params.numPoints,3))
        end = datetime.now()
        print("MMR Time:",(end-start).seconds)

        if network_params.dims==2:
            z = np.zeros((selectedSamples.shape[0],network_params.numPoints,1))
            selectedSamples = np.concatenate((selectedSamples,z),2)

        if training_params.project:
            #dumpSamples_top(misc_variables.hmcsampleDir,selectedSamples)

            bestShape = None
            perturbedShape = np.array(selectedSamples)[:,:,:3]
            perturbedShape = torch.from_numpy(perturbedShape).float().cuda()

            bestEnergy = 1e9
            prevEnergy = 1e9
            for i in range(training_params.optEpochs):
                energy = None
                if training_params.energy == 'arap' or training_params.energy == 'regarap' or training_params.energy == 'arap_anal':
                    energy,rotations = losses.arap(pts[:,:,:3],perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
                    #updated_rotations = losses.bending(perturbedShape,pts[:,:,:3],neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,rotations,network_params.testEnergyWeight)
                    rhs = losses.arap_solve_rhs(pts[:,:,:3],perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,rotations,network_params.testEnergyWeight)
                    #perturbedShape = torch.unsqueeze(torch.mm(torch.inverse(square_weightMatrix),rhs),0)
                    perturbedShape = torch.linalg.solve(square_weightMatrix,rhs)
                    perturbedShape = torch.unsqueeze(perturbedShape,0)
                    energy,_ = losses.arap(perturbedShape,pts[:,:,:3],neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
                    energy = energy.mean()

                if energy.item() < bestEnergy:
                    bestEnergy = energy.item()/network_params.testEnergyWeight.item()
                    bestShape = perturbedShape 

                if abs(bestEnergy) < training_params.bestenergy:
                    break
                
                prevEnergy = energy.item()
            bestShape_closed = bestShape.clone().detach()
            bestEnergy_closed = bestEnergy

            print("perturbed",i,bestEnergy,flush=True)
            if bestShape is None:
                continue

            if bestEnergy>training_params.bestenergy:
                bestEnergy = 1e9
                bestShape = None
                perturbedShape = np.array(selectedSamples)[:,:,:3]
                perturbedShape = torch.from_numpy(perturbedShape).float().cuda()
                proposedShape = header.Position(perturbedShape.view(1,-1))
                for parameter in network_params.autoEncoder.encoder.parameters():
                    parameter.requires_grad = False
                codeOptimizer = torch.optim.Adam(proposedShape.parameters(),lr=1e-3,weight_decay=0 )

                for i in range(training_params.optEpochs):
                    codeOptimizer.zero_grad()

                    energy = None
                    if training_params.energy == 'arap' or training_params.energy == 'arap_anal':
                        energy,_ = losses.arap(pts[:,:,:3],proposedShape.proposed.view(1,network_params.numPoints,3),neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
                        energy = energy.mean()

                    energy.backward()
                    codeOptimizer.step()
                    if energy.item() < bestEnergy:
                        bestEnergy = energy.item()/network_params.testEnergyWeight.item()
                        bestShape = proposedShape.proposed.view(1,network_params.numPoints,3)

                    if bestEnergy < training_params.bestenergy:
                        break

                for parameter in network_params.autoEncoder.encoder.parameters():
                    parameter.requires_grad = True
                print("perturbed optimizer",i,bestEnergy,flush=True)
            bestShape_opt = bestShape.clone().detach()
            bestEnergy_opt = bestEnergy

            bestShape = bestShape_opt
            if bestEnergy_opt>1e-3: # and bestEnergy_opt>1e-3:
                continue
            if bestEnergy_closed < bestEnergy_opt:
                bestShape = bestShape_closed
            else:
                bestShape = bestShape_opt

            selectedSamples = bestShape.cpu().detach().numpy()

            if network_params.dims==2:
                z = np.zeros((selectedSamples.shape[0],network_params.numPoints,1))
                selectedSamples = np.concatenate((selectedSamples,z),2)
            dumpSamples_top(misc_variables.sampleDir,selectedSamples)
        else:
            dumpSamples_top(misc_variables.sampleDir,selectedSamples)

    for parameter in network_params.autoEncoder.decoder1.parameters():
        parameter.requires_grad = False
    network_params.autoEncoder.train()
