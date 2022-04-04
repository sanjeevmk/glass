import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
from torch.autograd import Variable
from Utils import Pts
from datetime import datetime
import numpy as np
from RegArap import ARAP

def get_jacobian_rand(cur_shape, z, model, epsilon=[1e-3], nz_max=10):
    nb, nz = z.size()
    _, n_vert, nc = cur_shape.size()
    if nz >= nz_max:
      rand_idx = np.random.permutation(nz)[:nz_max]
      nz = nz_max
    else:
      rand_idx = np.arange(nz)
    
    jacobian = torch.zeros((nb, n_vert*nc, nz)).cuda()
    for i, idx in enumerate(rand_idx):
        dz = torch.zeros(z.size()).cuda()
        dz[:, idx] = epsilon
        z_new = z + dz
        shape_new = model(z_new)
        dout = (shape_new - cur_shape).view(nb, -1)
        jacobian[:, :, i] = dout/epsilon
    return jacobian

def trainVAE(training_params,data_classes,network_params,losses,misc_variables,rounds,scale):
    if rounds==0:
        epochs = training_params.startepochs
    else:
        epochs = training_params.epochs 

    #_,template_face,_,_,_,_,_,_,_,_ = data_classes.original[0]
    _,template_face,_ = data_classes.original[0]
    arapreg = ARAP(template_face,network_params.numPoints,norm=training_params.norm).cuda()
    use_arapreg = False
    prev_best_epoch = 0
    print("Epochs:", epochs)
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
        for ind,data in enumerate(data_classes.torch_expanded):
            network_params.optimizer.zero_grad()
            #pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,names,_ = data
            #pts,faces,numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,area,names,p_pts,_ = data
            pts,faces,names = data
            if pts.size()[0]==1:
                continue


            pts = pts[:,:,:network_params.dims]
            pts = pts.float().cuda()
            #p_pts = p_pts[:,:,:network_params.dims]
            #p_pts = p_pts.float().cuda()
            #area = area.float().cuda()
            faces = faces.int().cuda()
            #numNeighbors = numNeighbors.int().cuda()
            #accnumNeighbors = accnumNeighbors.int().cuda()
            #neighborsMatrix = neighborsMatrix.int().cuda()
            #weightMatrix = weightMatrix.float().cuda()

            #code = network_params.autoEncoder.encoder(torch.unsqueeze(pts,1))
            code = network_params.autoEncoder.encoder(pts.transpose(2,1))
            reconstruction = network_params.autoEncoder.decoder1(code.squeeze())

            reconstructionError = losses.mse(pts,reconstruction)

            if torch.isnan(reconstructionError):
                print("nan occured")
                return "nan error"

            regLoss = losses.regularizer(code.squeeze())

            energyLoss = torch.zeros(1,1).float().cuda()
            if training_params.energy=='arap' or training_params.energy == 'arap_anal':
                jacob = get_jacobian_rand(reconstruction, code,network_params.autoEncoder.decoder1,epsilon=1e-1,nz_max=32)
                energyLoss,_ = arapreg(reconstruction, jacob)
                if isinstance(energyLoss,str):
                    if energyLoss=="nan":
                        return "eig failed"

                energyLoss /= jacob.shape[-1]
                energyLoss = network_params.energyWeight*energyLoss
                #energyLoss,_ = losses.arap(pts,reconstruction,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.energyWeight)

            L = torch.zeros(1,1).float().cuda()
            if training_params.energy=='NoEnergy':
                L = reconstructionError + regLoss
            else:
                #if use_arapreg:
                L = reconstructionError + regLoss + energyLoss.mean()

            #L = reconstructionError
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
            regLoss = losses.regularizer(code.squeeze())

            #if not use_arapreg:
            #    if __reconstructionError.item()<1e-3:
            #        use_arapreg = True

            energyLoss = torch.zeros(1,1).float().cuda()
            if training_params.energy=='arap':
                energyLoss,_ = losses.arap(pts,reconstruction,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,network_params.energyWeight)
            if training_params.energy=='regarap': # and use_arapreg:
                jacob = get_jacobian_rand(reconstruction, code,network_params.autoEncoder.decoder1,epsilon=1e-1,nz_max=32)
                energyLoss = network_params.energyWeight*arapreg(reconstruction, jacob,) / jacob.shape[-1]

            L = torch.zeros(1,1).float().cuda()
            if training_params.energy=='NoEnergy':
                L = __reconstructionError + regLoss
            else:
                #if use_arapreg:
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
            prev_best_epoch = ep
            misc_variables.bestError = meanReconError
            torch.save(network_params.autoEncoder.state_dict(),network_params.weightPath+"_r"+str(rounds))
            print("Saved",misc_variables.bestError)
            if misc_variables.bestError<1e-7:
                break
            network_params.autoEncoder.eval()
            #network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))
        elif ep>0 and misc_variables.bestError < 1e-6 and ep-prev_best_epoch>=30:
            break
    print("Round "+str(rounds)+" Besht Error:"+str(misc_variables.bestError))
