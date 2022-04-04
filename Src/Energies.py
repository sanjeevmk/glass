import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import torch.nn as nn
import torch.autograd as autograd
import sys
sys.path.append("../")
from extension.arap.cuda.arap import Arap
from extension.grad_arap.cuda.arap import ArapGrad
from RegArap import ARAP as ArapReg 
#from extension.isometric.cuda.isometric import Isometric
#from extension.asap.cuda.asap import Asap
#from extension.arap2d.cuda.arap import Arap as Arap2D
#from extension.asap2d.cuda.asap import Asap as Asap2D
#from extension.carap.cuda.carap import CArap
#from extension.casap.cuda.casap import CAsap
#from extension.asapEigen.cuda.asapEigen import Asap as AsapEigen
#from extension.arapEigen.cuda.arapEigen import Arap as ArapEigen
#from extension.gaussian.gaussian import GaussianEnergy
#from extension.pairwisedistance.pairwisedistance import PairwiseDistances
import numpy as np

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

class NLLPairwiseDistanceEnergyInput(nn.Module):
    def __init__(self,xyz1,weight):
        super(NLLPairwiseDistanceEnergyInput, self).__init__()
        self.xyz1 = xyz1
        self.energy = PairwiseDistances()
        self.weight = weight

    def forward(self,decoder,shape):
        reconstruction = shape.view(self.xyz1.size())
        energy = self.energy(self.xyz1,reconstruction,self.weight)
        grad = torch.autograd.grad(energy,reconstruction)[0]
        return self.weight*energy,self.weight*grad.view(shape.size())

class NLLPairwiseDistanceEnergy(nn.Module):
    def __init__(self,xyz1,weight):
        super(NLLPairwiseDistanceEnergy, self).__init__()
        self.xyz1 = xyz1
        self.energy = PairwiseDistances()
        self.weight = weight

    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder(code)
        energy = self.energy(self.xyz1,reconstruction,self.weight)
        energy.backward(torch.ones(energy.size()).float().cuda())
        grad = self.weight*code.grad
        decoder.zero_grad()
        return self.weight*energy,grad

class LogPairwiseDistanceEnergy(nn.Module):
    def __init__(self,xyz1,weight,decoder):
        super(LogPairwiseDistanceEnergy, self).__init__()
        self.xyz1 = xyz1
        self.energy = PairwiseDistances()
        self.decoder = decoder
        self.weight = weight

    def forward(self,code):
        self.decoder.zero_grad()
        code = np.expand_dims(code,0)
        code = torch.tensor(code,requires_grad=True,device='cuda',dtype=torch.float)
        reconstruction = self.decoder(code)
        energy = -1.0*self.energy(self.xyz1,reconstruction,self.weight)
        energy.backward()
        return energy.item(),(-1.0*code.grad[0]).detach().cpu().numpy()

class GaussianHmcEnergy(nn.Module):
    def __init__(self,mu,covinv,weight):
        super(GaussianHmcEnergy, self).__init__()
        self.mu = mu 
        self.covinv = covinv
        self.weight = weight 
        self.energy = GaussianEnergy()

    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder(code)
        energy = self.energy(self.mu,self.covinv,reconstruction,self.weight).mean()
        energy.backward()
        return energy,code.grad

class CArapEnergy(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,alpha,area,arapWeight):
        super(CArapEnergy, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.arapWeight = arapWeight
        self.alpha = alpha
        self.area = area
        self.arap = CArap()

    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder(code)
        arapEnergy = self.arap(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,self.alpha,self.area,self.arapWeight)
        meanEnergy = torch.mean(torch.mean(arapEnergy,2),1)
        meanEnergy.backward(torch.ones(meanEnergy.size()).float().cuda())
        #arapEnergy.backward()
        return meanEnergy,code.grad

class AsapEnergy2D(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,asapWeight):
        super(AsapEnergy2D, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.asapWeight = asapWeight
        self.asap = Asap2D()

    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder(code)
        asapEnergy = self.asap(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,self.asapWeight)
        meanEnergy = torch.mean(torch.mean(asapEnergy,2),1)
        meanEnergy.backward(torch.ones(meanEnergy.size()).float().cuda())
        return meanEnergy,code.grad

class AsapEnergy(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,asapWeight):
        super(AsapEnergy, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.asapWeight = asapWeight
        self.asap = Asap()

    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder(code)
        asapEnergy = self.asap(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,self.asapWeight)
        meanEnergy = torch.mean(torch.mean(asapEnergy,2),1)
        meanEnergy.backward(torch.ones(meanEnergy.size()).float().cuda())
        return meanEnergy,code.grad

class ArapEnergy2D(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,arapWeight):
        super(ArapEnergy2D, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.arapWeight = arapWeight
        self.arap = Arap2D()

    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder(code)
        arapEnergy = self.arap(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,self.arapWeight)
        meanEnergy = torch.mean(torch.mean(arapEnergy,2),1)
        meanEnergy.backward(torch.ones(meanEnergy.size()).float().cuda())
        return meanEnergy,code.grad

class IsometricEnergy(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,isometricWeight):
        super(IsometricEnergy, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.isometricWeight = isometricWeight
        self.isometric = Isometric()

    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder.decoder1(code)
        reconstruction = decoder.decoder2(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.isometricWeight)
        isometricEnergy = self.isometric(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.isometricWeight)
        meanEnergy = torch.mean(torch.mean(isometricEnergy,1))
        meanEnergy.backward(torch.ones(meanEnergy.size()).float().cuda())
        return meanEnergy,code.grad

def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, create_graph=create_graph,retain_graph=True)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    jacobian_mat = torch.stack(jac).reshape(y.shape + x.shape)
    return torch.squeeze(jacobian_mat)

def jacobian_hessian(y, x):
    jac = jacobian(y, x, create_graph=True)
    hess = jacobian(jac,x)
    return jac,hess

class ArapRegHessian(nn.Module):
    def __init__(self,template_face,num_points,nz_max=32):
        super(ArapRegHessian, self).__init__()
        self.nz_max = nz_max
        self.arapreg = ArapReg(template_face,num_points).cuda()


    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder.decoder1(code)
        #reconstruction = decoder.decoder2(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.accnumNeighbors,self.weightMatrix,self.arapWeight)
        #decoder.zero_grad()
        jacob = get_jacobian_rand(reconstruction, code,decoder.decoder1,epsilon=1e-2,nz_max=self.nz_max)
        arapEnergy,hbars = self.arapreg(reconstruction, jacob,)
        arapEnergy /= jacob.shape[-1]
        hess = hbars[0]

        grad = jacobian(arapEnergy,code)
        return arapEnergy,grad,hess

class ArapEnergyHessianAnalytical(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,template_face,num_points,nz_max,arapWeight):
        super(ArapEnergyHessianAnalytical, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.accnumNeighbors = accnumNeighbors
        self.weightMatrix = weightMatrix
        self.arapWeight = arapWeight
        self.nz_max = nz_max
        self.arap = Arap()
        self.arapreg = ArapReg(template_face,num_points).cuda()


    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder.decoder1(code)
        #jacob = get_jacobian_rand(reconstruction, code,decoder.decoder1,epsilon=1e-1,nz_max=self.nz_max)
        #arapEnergy,hbars = self.arapreg(reconstruction, jacob,)
        arapEnergy,_ = self.arap(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.accnumNeighbors,self.weightMatrix,self.arapWeight)
        meanEnergy = torch.mean(torch.mean(arapEnergy,2),1)
        grad,hess= jacobian_hessian(meanEnergy,code)
        #grad = jacobian(meanEnergy,code)
        #hess = hbars[0]
        return hess

class ArapEnergyHessian(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,template_face,num_points,nz_max,arapWeight):
        super(ArapEnergyHessian, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.accnumNeighbors = accnumNeighbors
        self.weightMatrix = weightMatrix
        self.arapWeight = arapWeight
        self.nz_max = nz_max
        self.arap = Arap()
        self.arapreg = ArapReg(template_face,num_points).cuda()


    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder.decoder1(code)
        jacob = get_jacobian_rand(reconstruction, code,decoder.decoder1,epsilon=1e-1,nz_max=self.nz_max)
        arapEnergy,hbars = self.arapreg(reconstruction, jacob,)
        if isinstance(arapEnergy,str):
            return "nan","nan"
        #arapEnergy,_ = self.arap(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.accnumNeighbors,self.weightMatrix,self.arapWeight)
        #meanEnergy = torch.mean(torch.mean(arapEnergy,2),1)
        #grad,hess= jacobian_hessian(meanEnergy,code)
        #grad = jacobian(meanEnergy,code)
        hess = hbars[0]
        return arapEnergy,hess

class ArapEnergy(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,arapWeight):
        super(ArapEnergy, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.accnumNeighbors = accnumNeighbors
        self.weightMatrix = weightMatrix
        self.arapWeight = arapWeight
        self.arap = Arap()

    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder.decoder1(code)
        arapEnergy,_ = self.arap(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.accnumNeighbors,self.weightMatrix,self.arapWeight)
        meanEnergy = torch.mean(torch.mean(arapEnergy,2),1)
        meanEnergy.backward(torch.ones(meanEnergy.size()).float().cuda())
        return meanEnergy,code.grad

class LogCArapEnergy(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,alpha,area,arapWeight,decoder):
        super(LogCArapEnergy, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.arapWeight = arapWeight
        self.decoder = decoder
        self.alpha = alpha
        self.area = area
        self.arap = CArap()

    def forward(self,code):
        self.decoder.zero_grad()
        code = np.expand_dims(code,0)
        code = torch.tensor(code,requires_grad=True,device='cuda',dtype=torch.float)
        reconstruction = self.decoder(code)
        arapEnergy = self.arap(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,self.alpha,self.area,self.arapWeight)
        arapEnergy = arapEnergy.mean()
        arapEnergy.backward()
        return arapEnergy.item(),(code.grad[0]).detach().cpu().numpy()

class LogAsapEnergy(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,asapWeight,decoder):
        super(LogAsapEnergy, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.asapWeight = asapWeight
        self.decoder = decoder
        self.asap = Asap()

    def forward(self,code):
        self.decoder.zero_grad()
        code = np.expand_dims(code,0)
        code = torch.tensor(code,requires_grad=True,device='cuda',dtype=torch.float)
        reconstruction = self.decoder(code)
        asapEnergy = -1.0*self.asap(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,self.asapWeight)
        asapEnergy = asapEnergy.mean()
        asapEnergy.backward()
        return asapEnergy.item(),(-1.0*code.grad[0]).detach().cpu().numpy()

class LogArapEnergy(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,arapWeight,decoder):
        super(LogArapEnergy, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.arapWeight = arapWeight
        self.decoder = decoder
        self.arap = Arap()

    def forward(self,code):
        self.decoder.zero_grad()
        code = np.expand_dims(code,0)
        code = torch.tensor(code,requires_grad=True,device='cuda',dtype=torch.float)
        reconstruction = self.decoder(code)
        arapEnergy = -1.0*self.arap(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,self.arapWeight)
        arapEnergy = arapEnergy.mean()
        arapEnergy.backward()
        return arapEnergy.item(),(-1.0*code.grad[0]).detach().cpu().numpy()

class CAsapEnergy(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,alpha,area,asapWeight):
        super(CAsapEnergy, self).__init__()
        self.xyz1 = xyz1
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.asapWeight = asapWeight
        self.alpha = alpha
        self.area = area
        self.asap = CAsap()

    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder(code)
        asapEnergy = self.asap(self.xyz1,reconstruction,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,self.alpha,self.area,self.asapWeight).mean()
        asapEnergy.backward()
        return asapEnergy,code.grad

class ArapEigenEnergy(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,eigC,eigV,eigVT,nComp,arapWeight):
        super(ArapEigenEnergy, self).__init__()
        self.xyz1 = torch.zeros(xyz1.size()).float().cuda()
        self.xyz2 = torch.zeros(xyz1.size()).float().cuda()
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.arapWeight = arapWeight
        self.eigC = eigC
        self.eigV = eigV
        self.eigVT = eigVT
        self.nComp = nComp
        self.arap = ArapEigen()

    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder(code)
        arapEnergy = self.arap(self.xyz1,self.xyz2,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,reconstruction,self.eigC,self.eigV,self.eigVT,self.nComp,self.arapWeight).mean()
        arapEnergy.backward()
        return arapEnergy,code.grad

class AsapEigenEnergy(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,eigC,eigV,eigVT,nComp,asapWeight):
        super(AsapEigenEnergy, self).__init__()
        self.xyz1 = torch.zeros(xyz1.size()).float().cuda()
        self.xyz2 = torch.zeros(xyz1.size()).float().cuda()
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.asapWeight = asapWeight
        self.eigC = eigC
        self.eigV = eigV
        self.eigVT = eigVT
        self.nComp = nComp
        self.asap = AsapEigen()

    def forward(self,decoder,code):
        decoder.zero_grad()
        reconstruction = decoder(code)
        asapEnergy = self.asap(self.xyz1,self.xyz2,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,reconstruction,self.eigC,self.eigV,self.eigVT,self.nComp,self.asapWeight).mean()
        asapEnergy.backward()
        return asapEnergy,code.grad

class AsapEigenEnergy_InputDomain(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,eigC,eigV,eigVT,nComp,asapWeight):
        super(AsapEigenEnergy_InputDomain, self).__init__()
        self.xyz1 = xyz1
        self.xyz2 = self.xyz1.clone().detach()
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.asapWeight = asapWeight
        self.eigC = eigC
        self.eigV = eigV
        self.eigVT = eigVT
        self.nComp = nComp
        self.asap = AsapEigen()

    def forward(self,newSample):
        asapEnergy = self.asap(self.xyz1,self.xyz2,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,newSample,self.eigC,self.eigV,self.eigVT,self.nComp,self.asapWeight).mean()
        asapEnergy.backward()
        return asapEnergy,newSample.grad

class ArapEigenEnergy_InputDomain(nn.Module):
    def __init__(self,xyz1,neighborsMatrix,numNeighbors,weightMatrix,eigC,eigV,eigVT,nComp,arapWeight):
        super(ArapEigenEnergy_InputDomain, self).__init__()
        self.xyz1 = xyz1
        self.xyz2 = self.xyz1.clone().detach()
        self.neighborsMatrix = neighborsMatrix 
        self.numNeighbors = numNeighbors
        self.weightMatrix = weightMatrix
        self.arapWeight = arapWeight
        self.eigC = eigC
        self.eigV = eigV
        self.eigVT = eigVT
        self.nComp = nComp
        self.arap = ArapEigen()

    def forward(self,newSample):
        arapEnergy = self.arap(self.xyz1,self.xyz2,self.neighborsMatrix,self.numNeighbors,self.weightMatrix,newSample,self.eigC,self.eigV,self.eigVT,self.nComp,self.arapWeight).mean()
        arapEnergy.backward()
        return arapEnergy,newSample.grad
