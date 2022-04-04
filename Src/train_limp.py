import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
from torch.autograd import Variable
from Utils import Pts

def trainVAE(training_params,data_classes,network_params,losses,misc_variables,rounds,scale):
    for ep in range(training_params.epochs):
        network_params.autoEncoder.train()
        batchIter = 0
        meanError = 0
        meanReconError = 0
        meanKLDError = 0
        meanArapError = 0
        savePts = []
        saveFaces = []
        for ind,data in enumerate(data_classes.torch_pair):
            network_params.optimizer.zero_grad()
            pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,area,names,pts1,faces1,numNeighbors1,accnumNeighbors1,neighborsMatrix1,weightMatrix1,area1,names1 = data
            if pts.size()[0]==1:
                continue


            pts = pts[:,:,:network_params.dims]
            pts = pts.float().cuda()
            pts1 = pts1[:,:,:network_params.dims]
            pts1 = pts1.float().cuda()
            area = area.float().cuda()
            area1 = area1.float().cuda()
            faces = faces.int().cuda()
            faces1 = faces1.int().cuda()
            numNeighbors = numNeighbors.int().cuda()
            accnumNeighbors = accnumNeighbors.int().cuda()
            numNeighbors1 = numNeighbors1.int().cuda()
            neighborsMatrix = neighborsMatrix.int().cuda()
            neighborsMatrix1 = neighborsMatrix1.int().cuda()
            weightMatrix = weightMatrix.float().cuda()
            weightMatrix1 = weightMatrix1.float().cuda()

            code1 = network_params.autoEncoder.encoder(pts.transpose(2,1))
            code2 = network_params.autoEncoder.encoder(pts1.transpose(2,1))
            reconstruction1 = network_params.autoEncoder.decoder1(code1)
            reconstruction2 = network_params.autoEncoder.decoder1(code2)
            reconstructionError1 = losses.mse(pts,reconstruction1)
            reconstructionError2 = losses.mse(pts1,reconstruction2)

            network_params.alpha.uniform_()

            interpCode = (network_params.alpha[:code1.size()[0],:]*code1) + ((1-network_params.alpha[:code1.size()[0],:])*code2)
            reconstruction3 = network_params.autoEncoder.decoder1(interpCode)

            energyLoss1 = torch.zeros(1,1).float().cuda()
            energyLoss2 = torch.zeros(1,1).float().cuda()
            if training_params.energy=='arap':
                energyLoss1,_ = losses.arap(reconstruction1,reconstruction3,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.energyWeight)
                energyLoss2,_ = losses.arap(reconstruction2,reconstruction3,neighborsMatrix1,numNeighbors1,accnumNeighbors,weightMatrix1,network_params.energyWeight)
            
            energyLoss = energyLoss1 + energyLoss2

            L = reconstructionError1 + reconstructionError2 + energyLoss.mean()
            L.backward()

            meanError += L.item()
            meanReconError += 0.5*(reconstructionError1.item()+reconstructionError2.item())
            meanArapError += energyLoss.mean().item()

            network_params.optimizer.step()
            network_params.optimizer.zero_grad()
            if batchIter%1==0:
                print("global:",batchIter,len(data_classes.torch_pair),ep,meanError/(batchIter+1),meanReconError/(batchIter+1),meanArapError/(batchIter+1))
            batchIter+=1
        #energyWeight = min(maxWeight,energyWeight+arapRate)# arapRate
        meanError/=len(data_classes.torch_pair) 
        meanReconError/=len(data_classes.torch_pair) 
        if meanReconError<misc_variables.bestError:
            misc_variables.bestError = meanReconError
            torch.save(network_params.autoEncoder.state_dict(),network_params.weightPath+"_r"+str(rounds))
            print("Saved",misc_variables.bestError)
            if misc_variables.bestError<1e-5:
                break
