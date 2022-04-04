import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
from torch.autograd import Variable

def trainVAE(training_params,data_classes,network_params,losses,misc_variables,rounds,scale):
    network_params.autoEncoder.eval()
    network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))
    for parameter in network_params.autoEncoder.encoder.parameters():
        parameter.requires_grad = False
    network_params.corrEncoder.train()
    for ep in range(training_params.correpochs):
        if ep==500:
            break
        batchIter = 0
        meanError = 0
        meanReconError = 0
        meanKLDError = 0
        meanArapError = 0
        savePts = []
        saveFaces = []
        for ind,data in enumerate(data_classes.torch_corrdata):
            network_params.corroptimizer.zero_grad()
            pts,t_pts,_ = data

            pts = pts[:,:,:network_params.dims]
            pts = pts.float().cuda()
            t_pts = t_pts[:,:,:network_params.dims]
            t_pts = t_pts.float().cuda()
            p_code = network_params.corrEncoder(pts.transpose(2,1))
            t_code = network_params.autoEncoder.encoder(t_pts.transpose(2,1))
            p_recon = network_params.autoEncoder.decoder1(p_code.squeeze())
            reconstructionError = losses.mse(t_pts,p_recon)
            codeError = losses.mse(t_code,p_code)
            print("target",t_code[0][:10])
            print("predict",p_code[0][:10])
            print(reconstructionError.item())
            L = codeError
            L.backward()
            network_params.corroptimizer.step()
            network_params.corroptimizer.zero_grad()
            meanError += codeError.item()
            if batchIter%1==0:
                print("global:",batchIter,len(data_classes.torch_corrdata),ep,meanError/(batchIter+1))
            batchIter+=1

        #energyWeight = min(maxWeight,energyWeight+arapRate)# arapRate
        meanError/=len(data_classes.torch_corrdata) 
        if meanError<misc_variables.bestError:
            misc_variables.bestError = meanError
            torch.save(network_params.corrEncoder.state_dict(),network_params.corrweightPath+"_r"+str(rounds))
            print("Saved",misc_variables.bestError)
            if misc_variables.bestError<1e-6:
                break
