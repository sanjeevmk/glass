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
def randomExplore(training_params,data_classes,network_params,losses,misc_variables,rounds):
    for parameter in network_params.autoEncoder.decoder1.parameters():
        parameter.requires_grad = False
    for parameter in network_params.autoEncoder.decoder2.parameters():
        parameter.requires_grad = False
    shapes = []
    meshNames = []
    for ind,data in enumerate(data_classes.expanded):
        pts,faces,numNeighbors,neighborsMatrix,weightMatrix,area,names,_ = data
        pts = torch.unsqueeze(torch.from_numpy(pts).float().cuda(),0)
        pts = pts[:,:,:network_params.dims]
        meshNames.extend(names)
        shapes.extend(pts)

    names = meshNames[0]

    print("Round:",rounds," HMC",flush=True)
    network_params.autoEncoder.eval()
    network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))

    if not os.path.exists(misc_variables.sampleDir):
        os.makedirs(misc_variables.sampleDir)

    def dumpSamples(outDir,samples,p_pts=None,corner=-1):
        for i in range(len(samples)):
            objInstance = Obj.Obj("dummy.obj")
            objInstance.setVertices(samples[i])
            objInstance.setFaces(faces)
            objInstance.saveAs(os.path.join(outDir,str(rounds) + "_" + names.split(".")[0] + "_" + str(i)+".obj"))

            if p_pts is not None:
                ptsInstance = Pts.Pts("dummy.pts")
                ptsInstance.setVertices(p_pts)
                ptsInstance.saveAs(os.path.join(outDir,str(rounds) + "_" + names.split(".")[0] + "_" + str(i)+".pts"))

    for ind,data in enumerate(data_classes.expanded):
        samples = []
        energies = []
        print("HMC Source Data:",ind,len(data_classes.expanded),flush=True)
        pts,faces,numNeighbors,neighborsMatrix,weightMatrix,area,names,original_parent = data
        query = np.expand_dims(pts[:,:network_params.dims],0)
        query = np.reshape(query,(1,-1))
        pts = torch.unsqueeze(torch.from_numpy(pts).float().cuda(),0)
        pts = pts[:,:,:network_params.dims]
        p_pts = torch.unsqueeze(torch.from_numpy(original_parent).float().cuda(),0)
        p_pts = p_pts[:,:,:network_params.dims]
        numNeighbors = torch.unsqueeze(torch.from_numpy(numNeighbors).int().cuda(),0)
        neighborsMatrix = torch.unsqueeze(torch.from_numpy(neighborsMatrix).int().cuda(),0)
        weightMatrix = torch.unsqueeze(torch.from_numpy(weightMatrix).float().cuda(),0)
        area = torch.unsqueeze(torch.from_numpy(area).float().cuda(),0)

        energy_fn = None
        if training_params.energy=='pdist':
            energy_fn = Energies.NLLPairwiseDistanceEnergy(p_pts,network_params.testEnergyWeight)
        if training_params.energy=='arap':
            energy_fn = Energies.ArapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='arap2d':
            energy_fn = Energies.ArapEnergy2D(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='asap':
            energy_fn = Energies.AsapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='asap2d':
            energy_fn = Energies.AsapEnergy2D(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='carap':
            energy_fn = Energies.CArapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,misc_variables.alpha,area,network_params.testEnergyWeight)

        seedCode = network_params.autoEncoder.encoder(pts.transpose(2,1))
        recon = network_params.autoEncoder.decoder1(seedCode)
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(recon.cpu().detach().numpy()[0])
        objInstance.setFaces(faces)
        objInstance.saveAs(os.path.join(misc_variables.reconDir,str(rounds) + "_" + names.split(".")[0] + "_recon.obj"))

        network_params.noise.normal_()
        addedNoise = Variable(0.3*network_params.noise[:seedCode.size()[0],:])
        seedCode += addedNoise
        recon = network_params.autoEncoder.decoder1(seedCode)
        recon = network_params.autoEncoder.decoder2(pts,recon,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        samples = np.array([recon.cpu().detach().numpy()[0]])
        if network_params.dims==2:
            z = np.zeros((samples.shape[0],network_params.numPoints,1))
            samples = np.concatenate((samples,z),2)
        dumpSamples(misc_variables.hmcsampleDir,samples)

        stepSize = training_params.stepsize
        stepsizeMin = 0.001
        stepsizeMax = 0.1
        stepsizeInc = 1.02
        stepsizeDec = 0.98
        targetAcceptanceRate = 0.7
        avgAcceptanceSlowness = 0.9
        avgAcceptance = 0.0
        burnin = 0

        selectedSamples = np.expand_dims(np.array(samples[-1]),0)
        print(names)

        if network_params.dims==2:
            z = np.zeros((selectedSamples.shape[0],network_params.numPoints,1))
            selectedSamples = np.concatenate((selectedSamples,z),2)

        perturbedShape = np.array(selectedSamples)[:,:,:network_params.dims]
        perturbedShape = torch.from_numpy(perturbedShape).float().cuda()
        proposedShape = header.Position(perturbedShape.view(1,-1))
        codeOptimizer = torch.optim.Adam(proposedShape.parameters(),lr=1e-3,weight_decay=0 ) #1e-4)

        bestEnergy = 1e9
        prevEnergy = 1e9
        bestShape = None
        for i in range(training_params.optEpochs):
            codeOptimizer.zero_grad()

            energy = None
            if training_params.energy == 'pdist':
                energy = losses.pdist(p_pts,proposedShape.proposed.view(1,network_params.numPoints,network_params.dims),network_params.testEnergyWeight)
            if training_params.energy == 'arap':
                energy = losses.arap(pts,proposedShape.proposed.view(1,network_params.numPoints,network_params.dims),neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight).mean()
            if training_params.energy == 'arap2d':
                energy = losses.arap2d(pts,proposedShape.proposed.view(1,network_params.numPoints,network_params.dims),neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight).mean()
            if training_params.energy == 'asap':
                energy = losses.asap(pts,proposedShape.proposed.view(1,network_params.numPoints,network_params.dims),neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight).mean()
            if training_params.energy == 'asap2d':
                energy = losses.asap2d(pts,proposedShape.proposed.view(1,network_params.numPoints,network_params.dims),neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight).mean()
            if training_params.energy == 'carap':
                energy = losses.carap(pts,proposedShape.proposed.view(1,network_params.numPoints,network_params.dims),neighborsMatrix,numNeighbors,weightMatrix,misc_variables.alpha,area,network_params.testEnergyWeight).mean()

            energy.backward()
            codeOptimizer.step()
            if energy.item() < bestEnergy:
                bestEnergy = energy.item()/network_params.testEnergyWeight.item()
                bestShape = proposedShape.proposed.view(1,network_params.numPoints,network_params.dims)

            if bestEnergy < training_params.bestenergy:
                break
            
            prevEnergy = energy.item()
        print("perturbed",i,bestEnergy,flush=True)
        #if bestEnergy>training_params.bestenergy:
        #    continue

        selectedSamples = bestShape.cpu().detach().numpy()
        if network_params.dims==2:
            z = np.zeros((selectedSamples.shape[0],network_params.numPoints,1))
            selectedSamples = np.concatenate((selectedSamples,z),2)
        dumpSamples(misc_variables.sampleDir,selectedSamples,p_pts=original_parent)

    for parameter in network_params.autoEncoder.decoder1.parameters():
        parameter.requires_grad = False
    for parameter in network_params.autoEncoder.decoder2.parameters():
        parameter.requires_grad = False
