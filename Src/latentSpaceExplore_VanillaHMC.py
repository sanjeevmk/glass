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
def hmcExplore(training_params,data_classes,network_params,losses,misc_variables,rounds):
    for parameter in network_params.autoEncoder.decoder1.parameters():
        parameter.requires_grad = False
    shapes = []
    meshNames = []

    for ind,data in enumerate(data_classes.full_arap):
        pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,names,_ = data
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

    def dumpSamples(outDir,samples,corner=-1):
        for i in range(len(samples)):
            objInstance = Obj.Obj("dummy.obj")
            objInstance.setVertices(samples[i])
            objInstance.setFaces(faces)
            objInstance.saveAs(os.path.join(outDir,str(rounds) + "_" + names.split(".")[0] + "_" + str(i)+".obj"))

    hmc = Leapfrog.Leapfrog(training_params.numsteps,0.1)

    for ind,data in enumerate(data_classes.full_arap):
        samples = []
        energies = []
        print("HMC Source Data:",ind,len(data_classes.full_arap),flush=True)
        pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,names,_ = data
        query = np.expand_dims(pts[:,:network_params.dims],0)
        query = np.reshape(query,(1,-1))
        pts = torch.unsqueeze(torch.from_numpy(pts).float().cuda(),0)
        pts = pts[:,:,:network_params.dims]
        numNeighbors = torch.unsqueeze(torch.from_numpy(numNeighbors).int().cuda(),0)
        accnumNeighbors = torch.unsqueeze(torch.from_numpy(accnumNeighbors).int().cuda(),0)
        neighborsMatrix = torch.unsqueeze(torch.from_numpy(neighborsMatrix).int().cuda(),0)
        weightMatrix = torch.unsqueeze(torch.from_numpy(weightMatrix).float().cuda(),0)

        energy_fn = None
        if training_params.energy=='pdist':
            energy_fn = Energies.NLLPairwiseDistanceEnergy(p_pts,network_params.testEnergyWeight)
        if training_params.energy=='arap':
            energy_fn = Energies.ArapEnergy(pts,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='arap2d':
            energy_fn = Energies.ArapEnergy2D(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='asap':
            energy_fn = Energies.AsapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='asap2d':
            energy_fn = Energies.AsapEnergy2D(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)
        if training_params.energy=='carap':
            energy_fn = Energies.CArapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,misc_variables.alpha,area,network_params.testEnergyWeight)
        #seedCode = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
        seedCode = network_params.autoEncoder.encoder(pts.transpose(2,1))
        #network_params.noise.normal_()
        #addedNoise = Variable(0.5*network_params.noise[:seedCode.size()[0],:])
        #seedCode += addedNoise

        recon = network_params.autoEncoder.decoder1(seedCode)
        #recon = network_params.autoEncoder.decoder2(pts,recon,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight)

        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(recon.cpu().detach().numpy()[0])
        objInstance.setFaces(faces)
        #dumpSamples(misc_variables.sampleDir,selectedSamples,p_pts=original_parent)
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
        for i in range(training_params.hmcEpochs):
            print(i)
            newCode,energy,accept = hmc(code,energy_fn,stepSize=stepSize,decoder=network_params.autoEncoder)

            newSample = network_params.autoEncoder.decoder1(newCode)

            network_params.autoEncoder.zero_grad()
            code = newCode.clone().detach()
            if network_params.dims == 2:
                z = torch.zeros(1,network_params.numPoints,1).float().cuda()
                newSample = torch.cat((newSample,z),2)
            if accept>0: # and energy<hmcThresh:
                print("accepted")
                energies.append(energy/network_params.testEnergyWeight.item())
                samples.append(newSample.cpu().detach().numpy()[0])
        end = datetime.now()
        print("HMC Time:",(end-start).seconds)
        if len(energies) == 0:
            continue
        print("min energy:",min(energies))
        if len(samples)==0:
            continue

        start = datetime.now()

        selectedSamples = np.array(samples)
        print(names)
        dumpSamples(misc_variables.sampleDir,selectedSamples)

    for parameter in network_params.autoEncoder.decoder1.parameters():
        parameter.requires_grad = False
