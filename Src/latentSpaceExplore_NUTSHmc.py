import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs

import Energies
import Leapfrog
from Utils import writer,Off,Obj,Pts
import header
from torch.autograd import Variable

def nutsExplore(training_params,data_classes,network_params,losses,misc_variables,rounds):
    for parameter in network_params.autoEncoder.decoder.parameters():
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

    def dumpSamples(outDir,samples,p_pts=None):
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

        if training_params.energy == 'pdist':
            energy_fn = Energies.LogPairwiseDistanceEnergy(p_pts,network_params.testEnergyWeight,network_params.autoEncoder.decoder)
        if training_params.energy=='arap':
            energy_fn = Energies.LogArapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,network_params.testEnergyWeight,network_params.autoEncoder.decoder)
        #arapEnergy = ArapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,testEnergyWeight)
        #arapEnergy = CArapEnergy(pts,neighborsMatrix,numNeighbors,weightMatrix,alpha,area,testEnergyWeight)
        #code = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
        code = network_params.autoEncoder.encoder(pts.transpose(2,1))
        #network_params.noise.normal_()
        #addedNoise = Variable(1.0*network_params.noise[:code.size()[0],:])
        #code += addedNoise
        recon = network_params.autoEncoder.decoder(code)
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(recon.cpu().detach().numpy()[0])
        objInstance.setFaces(faces)
        objInstance.saveAs(os.path.join(misc_variables.sampleDir,str(rounds) + "_" + names.split(".")[0] + "_recon" + ".obj"))

        burnin=0
        stepSize  = 0.1
        codeSamples,energy,_ = header.nuts6(stepSize,energy_fn.forward,training_params.hmcEpochs,burnin,code.detach().cpu().numpy()[0],0.6)
        if len(codeSamples)<2:
            continue
        codeSamples = codeSamples[1:,:]
        energy = energy[1:]
        energy *= -1
        energy /= network_params.testEnergyWeight.item()
        print(len(codeSamples),np.min(energy))
        if len(codeSamples)==0:
            continue
        codeSamples = torch.tensor(codeSamples,device='cuda',dtype=torch.float)

        samples = torch.zeros((1,network_params.numPoints,network_params.dims)).float().cuda(0)

        for bind in range(0,len(codeSamples),training_params.batch):
            shapeSamples = network_params.autoEncoder.decoder(codeSamples[bind:bind+training_params.batch,:])
            samples = torch.cat([samples,shapeSamples],0)
        
        samples = samples[1:,:,:]
        samples = samples.detach().cpu().numpy()

        selectedSamples = np.array([s.cpu().detach().numpy() for i,s in enumerate(shapes) if meshNames[i]!=names])
        selectedSamples = np.reshape(selectedSamples,(selectedSamples.shape[0],-1))
        initialLen = len(selectedSamples)
        samples = np.array(samples)

        mmr_lambda=0.3
        print("Round:",rounds,len(samples)," SAMPLE SELECTION")
        if len(samples)>0:
            for numAddedSamples in range(misc_variables.numNewSamples):
                print("sample adding:",numAddedSamples,misc_variables.numNewSamples)
                allDistances = []
                for s in samples:
                    flatS = np.reshape(s[:,:network_params.dims],(1,-1))
                    QDist = np.mean((query- flatS)**2,1)[0]
                    SSimilarity = np.max(cs(selectedSamples,flatS))
                    QSimilarity = np.max(cs(query,flatS))
                    similarity = mmr_lambda*QSimilarity - (1-mmr_lambda)*SSimilarity
                    allDistances.append(similarity)

                allDistances = np.array(allDistances)
                index = np.argmax(allDistances)
                selectedS = samples[index,:,:]
                samples = np.delete(samples,index,axis=0)
                selectedSamples = np.vstack((selectedSamples,np.reshape(selectedS[:,:network_params.dims],(1,-1))))

        selectedSamples = selectedSamples[initialLen:,:]
        selectedSamples = np.reshape(selectedSamples,(selectedSamples.shape[0],network_params.numPoints,network_params.dims))
        if network_params.dims==2:
            z = np.zeros((selectedSamples.shape[0],network_params.numPoints,1))
            selectedSamples = np.concatenate((selectedSamples,z),2)
        dumpSamples(misc_variables.hmcsampleDir,selectedSamples)

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
            #energy = arap(pts,proposedShape,neighborsMatrix,numNeighbors,weightMatrix,testEnergyWeight).mean()
            #energy = carap(pts,proposedShape,neighborsMatrix,numNeighbors,weightMatrix,alpha,area,testEnergyWeight).mean()
            energy.backward()
            codeOptimizer.step()

            if energy.item() < bestEnergy:
                bestEnergy = energy.item()/network_params.testEnergyWeight.item()
                bestShape = proposedShape.proposed.view(1,network_params.numPoints,network_params.dims)

            if bestEnergy < training_params.bestenergy:
                break
            
            prevEnergy = energy.item()
        print("perturbed",bestEnergy,flush=True)
        if bestEnergy>training_params.bestenergy:
            continue

        selectedSamples = bestShape.cpu().detach().numpy()
        if network_params.dims==2:
            z = np.zeros((selectedSamples.shape[0],network_params.numPoints,1))
            selectedSamples = np.concatenate((selectedSamples,z),2)
        dumpSamples(misc_variables.sampleDir,selectedSamples,p_pts=original_parent)
