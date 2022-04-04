import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
from torch.autograd import Variable
from Utils import Pts
from datetime import datetime

def trainVAE(training_params,data_classes,network_params,losses,misc_variables,rounds,scale):
    if rounds==0:
        epochs = training_params.startepochs
    else:
        epochs = training_params.epochs 
    for ep in range(epochs):
        network_params.autoEncoder.train()
        batchIter = 0
        meanError = 0
        meanReconError = 0
        meanKLDError = 0
        meanArapError = 0
        savePts = []
        saveFaces = []
        epStart = datetime.now()
        for ind,data in enumerate(data_classes.torch_full_arap):
            network_params.optimizer.zero_grad()
            pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,names,_ = data
            if pts.size()[0]==1:
                continue


            pts = pts[:,:,:network_params.dims]
            pts = pts.float().cuda()
            #p_pts = p_pts[:,:,:network_params.dims]
            #p_pts = p_pts.float().cuda()
            #area = area.float().cuda()
            faces = faces.int().cuda()
            numNeighbors = numNeighbors.int().cuda()
            accnumNeighbors = accnumNeighbors.int().cuda()
            neighborsMatrix = neighborsMatrix.int().cuda()
            weightMatrix = weightMatrix.float().cuda()

            #code = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
            code = network_params.autoEncoder.encoder(pts.transpose(2,1))
            reconstruction = network_params.autoEncoder.decoder1(code.squeeze())

            reconstructionError = losses.mse(pts,reconstruction)
            regLoss = losses.regularizer(code.squeeze())
            if training_params.energy=='arap':
                energyLoss,_ = losses.arap(pts,reconstruction,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.energyWeight)

            L = reconstructionError + regLoss + energyLoss.mean()
            L.backward()
            network_params.optimizer.step()
            network_params.optimizer.zero_grad()

            '''
            code = network_params.autoEncoder.encoder(pts.transpose(2,1))
            #network_params.noise.normal_()
            #addedNoise = Variable(scale*network_params.noise[:code.size()[0],:])
            #code += addedNoise

            reconstruction = network_params.autoEncoder.decoder1(code.squeeze())

            __reconstructionError = losses.mse(pts,reconstruction)


            energyLoss = torch.zeros(1,1).float().cuda()
            if training_params.energy=='pdist':
                energyLoss = losses.pdist(pts,reconstruction,network_params.energyWeight)
            if training_params.energy=='arap':
                energyLoss,_ = losses.arap(pts,reconstruction,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.energyWeight)
            if training_params.energy=='iso':
                energyLoss = losses.isometric(pts,reconstruction,neighborsMatrix,numNeighbors,network_params.energyWeight)
            if training_params.energy=='arap2d':
                energyLoss = losses.arap2d(pts,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,network_params.energyWeight)
            if training_params.energy=='asap':
                energyLoss = losses.asap(pts,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,network_params.energyWeight)
            if training_params.energy=='asap2d':
                energyLoss = losses.asap2d(pts,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,network_params.energyWeight)

            L = torch.zeros(1,1).float().cuda()
            if training_params.energy=='NoEnergy':
                L = __reconstructionError + regLoss
            else:
                L = __reconstructionError + regLoss + energyLoss.mean()

            L.backward() #gradient=torch.ones(training_params.batch,network_params.numPoints).float().cuda())
            network_params.optimizer.step()
            network_params.optimizer.zero_grad()
            '''

            meanError += L.item()
            meanReconError += reconstructionError.item()
            meanKLDError += regLoss.item()
            meanArapError += energyLoss.mean().item()
            savePts = reconstruction.clone().detach()
            if batchIter%1==0:
                print("global:",batchIter,len(data_classes.torch_expanded),ep,meanError/(batchIter+1),meanReconError/(batchIter+1),meanKLDError/(batchIter+1),meanArapError/(batchIter+1))
            batchIter+=1
        meanError/=len(data_classes.torch_expanded) 
        meanReconError/=len(data_classes.torch_expanded) 
        epEnd = datetime.now()
        print("Epoch duration:",(epEnd-epStart).seconds,flush=True)
        if meanReconError<misc_variables.bestError:
            misc_variables.bestError = meanReconError
            torch.save(network_params.autoEncoder.state_dict(),network_params.weightPath+"_r"+str(rounds))
            print("Saved",misc_variables.bestError)
            if misc_variables.bestError<1e-5:
                break
            network_params.autoEncoder.eval()
            network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))
    print("Round "+str(rounds)+" Besht Error:"+str(misc_variables.bestError))
