import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)

def randomSample(training_params,data_classes,network_params,losses,misc_variables,rounds):
    samples = []
    meshNames = []
    meshFaces = []
    shapes = []
    allnumNeighbors = []
    allneighborsMatrix = []
    allweightMatrix = []
    allFaces = []
    allAreas = []
    network_params.autoEncoder.eval()
    network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))
    for ind,data in enumerate(data_classes.torch_original):
        pts,faces,numNeighbors,neighborsMatrix,weightMatrix,area,names,_ = data
        pts = pts[:,:,:network_params.dims]
        pts = pts.float().cuda()
        faces = faces.int().cuda()
        area = area.float().cuda()
        numNeighbors = numNeighbors.int().cuda()
        neighborsMatrix = neighborsMatrix.int().cuda()
        weightMatrix = weightMatrix.float().cuda()

        code = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
        samples.extend(torch.unsqueeze(code,1))
        meshNames.extend(names)
        shapes.extend(pts)
        allnumNeighbors.extend(numNeighbors)
        allneighborsMatrix.extend(neighborsMatrix)
        allweightMatrix.extend(weightMatrix)
        allAreas.extend(area)
        allFaces.extend(faces)
        meshFaces = faces[0]

    print("Round:",rounds," SAMPLING",flush=True)
    samplingEnergyScores = []
    samplesTensor = torch.cat(samples,0)
    for i in range(training_params.samplingRounds):
        rand_sample = torch.randn(1,network_params.bottleneck).float().cuda()
        rand_out = network_params.autoEncoder.decoder(rand_sample)
        avgEnergy=0.0
        for j in range(len(shapes)):
            if training_params.energy=="pdist":
                energyLoss = losses.pdist(torch.unsqueeze(shapes[j],0),rand_out,network_params.testEnergyWeight).mean()
            if training_params.energy=="arap":
                energyLoss = losses.arap(torch.unsqueeze(shapes[j],0),rand_out,torch.unsqueeze(allneighborsMatrix[j],0),torch.unsqueeze(allnumNeighbors[j],0),torch.unsqueeze(allweightMatrix[j],0),network_params.testEnergyWeight).mean()
            #energyLoss = carap(torch.unsqueeze(shapes[j],0),rand_out,torch.unsqueeze(allneighborsMatrix[j],0),torch.unsqueeze(allnumNeighbors[j],0),torch.unsqueeze(allweightMatrix[j],0),alpha,torch.unsqueeze(allAreas[j],0),testEnergyWeight).mean()
            avgEnergy += (energyLoss.item()/network_params.testEnergyWeight.item())
        avgEnergy/=len(shapes)
        samplingEnergyScores.append(avgEnergy)
    meanSampledEnergy = sum(samplingEnergyScores)/len(samplingEnergyScores)
    with open(misc_variables.plotFile,'a+') as f:
        f.write(str(meanSampledEnergy)+"\n")
