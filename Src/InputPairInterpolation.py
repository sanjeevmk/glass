import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs

import Energies
import Leapfrog
import Tangent 
from Utils import writer,Off,Obj,Pts,rect_remesh
import header 
from torch.autograd import Variable
def pairInterpolate(training_params,data_classes,network_params,losses,misc_variables,rounds):
    if not os.path.exists(misc_variables.sampleDir):
        os.makedirs(misc_variables.sampleDir)

    def dumpSamples(outDir,samples,name):
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(samples[0])
        objInstance.setFaces(faces)
        objInstance.saveAs(os.path.join(outDir,name))

    pairs = []
    for ind,data in enumerate(data_classes.full_arap):
        print("Pair Src:",ind,len(data_classes.full_arap),flush=True)
        pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,names,square_weightMatrix = data
        pts = torch.unsqueeze(torch.from_numpy(pts).float().cuda(),0)
        pts = pts[:,:,:network_params.dims]
        numNeighbors = torch.unsqueeze(torch.from_numpy(numNeighbors).int().cuda(),0)
        accnumNeighbors = torch.unsqueeze(torch.from_numpy(accnumNeighbors).int().cuda(),0)
        neighborsMatrix = torch.unsqueeze(torch.from_numpy(neighborsMatrix).int().cuda(),0)
        weightMatrix = torch.unsqueeze(torch.from_numpy(weightMatrix).float().cuda(),0)
        square_weightMatrix = -1*torch.unsqueeze(torch.from_numpy(square_weightMatrix).float().cuda(),0)

        num_pairs = len(data_classes.full_arap)*(len(data_classes.full_arap)-1)/2
        target_num_shapes = misc_variables.num_shapes 
        per_pair_shapes =  target_num_shapes/num_pairs
        interval = float(1.0/per_pair_shapes)

        interval = 0.01

        for ind2,data2 in enumerate(data_classes.full_arap):
            if ind==ind2:
                continue
            if (ind2,ind) in pairs:
                continue
            print("Pair Tgt:",ind2,len(data_classes.full_arap),flush=True)
            pts2,faces2,numNeighbors2,accnumNeighbors2,neighborsMatrix2,weightMatrix2,names2,square_weightMatrix2 = data2
            pts2 = torch.unsqueeze(torch.from_numpy(pts2).float().cuda(),0)
            pts2 = pts2[:,:,:network_params.dims]
            numNeighbors2 = torch.unsqueeze(torch.from_numpy(numNeighbors2).int().cuda(),0)
            accnumNeighbors2 = torch.unsqueeze(torch.from_numpy(accnumNeighbors2).int().cuda(),0)
            neighborsMatrix2 = torch.unsqueeze(torch.from_numpy(neighborsMatrix2).int().cuda(),0)
            weightMatrix2 = torch.unsqueeze(torch.from_numpy(weightMatrix2).float().cuda(),0)
            square_weightMatrix2 = -1*torch.unsqueeze(torch.from_numpy(square_weightMatrix2).float().cuda(),0)

            for t in np.arange(0.0,1+interval,interval):
                print("At interpolation:",t,flush=True)
                perturbedShape = (1-t) * pts + t*pts2

                if training_params.energy == 'NoEnergy':
                    dumpSamples(misc_variables.sampleDir,perturbedShape.cpu().detach().numpy(),names.split(".")[0] + '_' + names2.split(".")[0] + '_' + "{:1.5f}".format(t) + '.obj')
                    continue

                #proposedShape = header.Position(perturbedShape.view(1,-1))
                #codeOptimizer = torch.optim.Adam(proposedShape.parameters(),lr=1e-3,weight_decay=0 ) #1e-4)

                bestEnergy = 1e9
                bestShape = None
                prevEnergy = None
                for x in range(training_params.optEpochs):
                    #codeOptimizer.zero_grad()

                    energy = None
                    if training_params.energy == 'arap':
                        #energy1,_ = losses.arap(pts,proposedShape.proposed.view(1,network_params.numPoints,network_params.dims),neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight) #.mean()
                        #energy2,_ = losses.arap(pts2,proposedShape.proposed.view(1,network_params.numPoints,network_params.dims),neighborsMatrix2,numNeighbors2,accnumNeighbors2,weightMatrix2,network_params.testEnergyWeight) #.mean()
                        #energy1 = energy1.mean()
                        #energy2 = energy2.mean()
                        #if energy1.item() < energy2.item():
                        #    energy = energy1
                        #else:
                        #    energy = energy2
                        #energy = energy1.mean() + energy2.mean()
                        if t<=0.5:
                            energy,rotations = losses.arap(pts[:,:,:3],perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
                            rhs = losses.arap_solve_rhs(pts[:,:,:3],perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,rotations,network_params.testEnergyWeight)
                            perturbedShape = torch.linalg.solve(square_weightMatrix,rhs)
                            energy,_ = losses.arap(pts[:,:,:3],perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
                            #energy,_ = losses.arap(pts,proposedShape.proposed.view(1,network_params.numPoints,network_params.dims),neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight) #.mean()
                            energy = energy.mean()
                        else:
                            energy,rotations = losses.arap(pts2[:,:,:3],perturbedShape,neighborsMatrix2,numNeighbors2,accnumNeighbors2,weightMatrix2,network_params.testEnergyWeight)
                            rhs = losses.arap_solve_rhs(pts2[:,:,:3],perturbedShape,neighborsMatrix2,numNeighbors2,accnumNeighbors2,weightMatrix2,rotations,network_params.testEnergyWeight)
                            perturbedShape = torch.linalg.solve(square_weightMatrix2,rhs)
                            energy,_ = losses.arap(pts2[:,:,:3],perturbedShape,neighborsMatrix2,numNeighbors2,accnumNeighbors2,weightMatrix2,network_params.testEnergyWeight)
                            #energy,_ = losses.arap(pts2,proposedShape.proposed.view(1,network_params.numPoints,network_params.dims),neighborsMatrix2,numNeighbors2,accnumNeighbors2,weightMatrix2,network_params.testEnergyWeight) #.mean()
                            #energy = energy.mean()
                            energy = energy.mean()

                    #energy.backward()
                    #codeOptimizer.step()
                    if energy.item() < bestEnergy:
                        bestEnergy = energy.item()/network_params.testEnergyWeight.item()
                        bestShape = perturbedShape

                    if bestEnergy < training_params.bestenergy:
                        break
                    
                    prevEnergy = energy.item()
                print("perturbed",x,bestEnergy,flush=True)
                if bestShape is None:
                    continue
                #if bestEnergy>training_params.bestenergy:
                #    continue

                selectedSamples = bestShape.cpu().detach().numpy()
                if network_params.dims==2:
                    z = np.zeros((selectedSamples.shape[0],network_params.numPoints,1))
                    selectedSamples = np.concatenate((selectedSamples,z),2)
                dumpSamples(misc_variables.sampleDir,selectedSamples,names.split(".")[0] + '_' + names2.split(".")[0] + '_' + "{:1.5f}".format(t) + '.obj')
            pairs.append((ind,ind2))
