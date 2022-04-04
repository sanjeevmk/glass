import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
from torch.autograd import Variable

def trainVAE(training_params,data_classes,network_params,losses,misc_variables,rounds):
    network_params.autoEncoder.train()
    for ep in range(training_params.epochs):
        batchIter = 0
        meanError = 0
        meanReconError = 0
        meanKLDError = 0
        meanArapError = 0
        savePts = []
        saveFaces = []
        for ind,data in enumerate(data_classes.torch_expanded):
            network_params.optimizer.zero_grad()
            pts,faces,numNeighbors,neighborsMatrix,weightMatrix,area,names = data
            pts = pts[:,:,:network_params.dims]
            pts = pts.float().cuda()
            area = area.float().cuda()
            faces = faces.int().cuda()
            numNeighbors = numNeighbors.int().cuda()
            neighborsMatrix = neighborsMatrix.int().cuda()
            weightMatrix = weightMatrix.float().cuda()
            code = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
            #noise.normal_()
            #addedNoise = Variable(0.01*noise[:code.size()[0],:])
            #code += addedNoise
            reconstruction = network_params.autoEncoder.decoder(code)

            #regLoss = regularizer(code)
            reconstructionError = losses.mse(pts,reconstruction)
            #energyLoss = pdist(pts,reconstruction,energyWeight)
            #energyLoss = arap(pts,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,energyWeight)
            #energyLoss = carap(pts,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,alpha,area,energyWeight)
            L = reconstructionError #+ regLoss + energyLoss.mean()
            L.backward()
            network_params.optimizer.step()
            network_params.optimizer.zero_grad()

            code = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
            network_params.noise.normal_()
            addedNoise = Variable(0.01*network_params.noise[:code.size()[0],:])
            code += addedNoise
            reconstruction = network_params.autoEncoder.decoder(code)
            regLoss = losses.regularizer(code)
            reconstructionError = losses.mse(pts,reconstruction)

            energyLoss = torch.zeros(1,1).float().cuda()
            if training_params.energy=='pdist':
                energyLoss = losses.pdist(pts,reconstruction,network_params.energyWeight)

            L = torch.zeros(1,1).float().cuda()
            if training_params.energy=='NoEnergy':
                L = reconstructionError + regLoss
            else:
                L = reconstructionError + regLoss + energyLoss.mean()
            L.backward()
            network_params.optimizer.step()
            network_params.optimizer.zero_grad()

            meanError += L.item()
            meanReconError += reconstructionError.item()
            meanKLDError += regLoss.item()
            meanArapError += energyLoss.mean().item()
            savePts = reconstruction.clone().detach()
            if batchIter%1==0:
                print("global:",batchIter,len(data_classes.torch_expanded),ep,meanError/(batchIter+1),meanReconError/(batchIter+1),meanKLDError/(batchIter+1),meanArapError/(batchIter+1))
            batchIter+=1

        #energyWeight = min(maxWeight,energyWeight+arapRate)# arapRate
        meanError/=len(data_classes.torch_expanded) 
        meanReconError/=len(data_classes.torch_expanded) 
        if meanReconError<misc_variables.bestError:
            misc_variables.bestError = meanReconError
            network_params.autoEncoder.eval()
            testnoise = torch.FloatTensor(training_params.batch, network_params.bottleneck).requires_grad_(False).cuda()
            testMeanReconError=0.0
            for ind,data in enumerate(data_classes.torch_test):
                pts,faces,names = data
                pts = pts[:,:,:network_params.dims]
                pts = pts.float().cuda()
                code = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
                testnoise.normal_()
                addedNoise = Variable(0.01*testnoise[:code.size()[0],:])
                code += addedNoise
                reconstruction = network_params.autoEncoder.decoder(code)
                reconstructionError = losses.mse(pts,reconstruction)

                testMeanReconError += reconstructionError.item()
            testMeanReconError/=len(data_classes.torch_test)
            network_params.autoEncoder.train()
            if testMeanReconError<misc_variables.testBestError:
                misc_variables.testBestError = testMeanReconError
                torch.save(network_params.autoEncoder.state_dict(),network_params.weightPath+"_r"+str(rounds))
                print("Saved",misc_variables.testBestError)
                if misc_variables.testBestError<0.0000001:
                    break