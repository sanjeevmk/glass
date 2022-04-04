import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import torch.nn as nn
import random
from torch.autograd import Variable
import time
class Leapfrog(nn.Module):
    def __init__(self,numSteps,stepSize,**kwargs):
        super(Leapfrog, self).__init__()
        self.numSteps = numSteps 
        self.stepSize = stepSize
        self.kwargs = kwargs
        #self.one = torch.Tensor([1.0])
        self.one = Variable(torch.tensor(1.0).cuda())

    def forward(self,previousSample,energy_function,numSteps=1,stepSize=0.01,**kwargs):
        new_draw = previousSample.clone().detach().requires_grad_(True)
        Mntm = torch.randn_like(new_draw)
        prev_U,gradient = energy_function(kwargs["decoder"],new_draw)
        #prev_K = torch.mm(mntm[0,:],mntm[0,:])/2.0
        prev_K = torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
        Mntm -= stepSize*gradient/2.0
        new_draw = (new_draw-stepSize * Mntm).clone().detach().requires_grad_(True)
        current_U,gradient = energy_function(kwargs["decoder"],new_draw)
        current_K= torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
        for _ in range(1,self.numSteps):
            Mntm -= stepSize*gradient
            new_draw = (new_draw-stepSize * Mntm).clone().detach().requires_grad_(True)
            current_U,gradient = energy_function(kwargs["decoder"],new_draw)
            current_K= torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
        Mntm -= stepSize*gradient/2.0
        Mntm *= -1
        #current_U,gradient = energy_function(kwargs["decoder"],new_draw)
        current_K= torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
        Z = torch.rand(current_K.size()[0]).cuda()
        criterion = torch.min(self.one,torch.exp(prev_U+prev_K-current_U-current_K))
        draw_final = new_draw.clone().detach()
        draw_final[Z>=criterion] = (previousSample.clone().detach())[Z>=criterion]
        acceptance = (Z<criterion).float()

        U_final = current_U.clone().detach()
        U_final[Z>=criterion] = prev_U[Z>=criterion]

        return draw_final,U_final,acceptance

    def forward1Step(self,previousSample,mntm,energy_function,energy_gradient_function,numSteps=1,**kwargs):
        new_draw = previousSample.clone().detach().requires_grad_(True)
        Mntm = mntm.clone().detach()
        kwargs["decoder"].eval()
        prev_U = energy_function(kwargs["decoder"],new_draw,Mntm,kwargs["NSi"],kwargs["Si"])
        prev_K = torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
        grad = energy_gradient_function(kwargs["decoder"],new_draw,Mntm,kwargs["NSi"],kwargs["Si"])
        Mntm -= self.stepSize*grad/2.0
        new_draw = (new_draw+self.stepSize * Mntm).clone().detach().requires_grad_(True)
        Mntm -= self.stepSize*energy_gradient_function(kwargs["decoder"],new_draw,Mntm,kwargs["NSi"],kwargs["Si"])/2.0
        Mntm *= -1
        current_U = energy_function(kwargs["decoder"],new_draw,Mntm,kwargs["NSi"],kwargs["Si"])
        current_K= torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
        Z = torch.rand(self.batch)
        criterion = torch.min(self.one,torch.exp(prev_U+prev_K-current_U-current_K))
        draw_final = new_draw.clone().detach()
        draw_final[Z>=criterion] = (previousSample.clone().detach())[Z>=criterion]

        kwargs["decoder"].train()

        return draw_final,grad

    def forwardTest(self,previousSample,mntm,energy_function,energy_gradient_function,numSteps=1,**kwargs):
        new_draw = previousSample.clone().detach().requires_grad_(True)
        Mntm = mntm.clone().detach()
        prev_U = energy_function(kwargs["decoder"],new_draw,Mntm,kwargs["NSi"],kwargs["Si"])
        prev_K = torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
        grad = energy_gradient_function(kwargs["decoder"],new_draw,Mntm,kwargs["NSi"],kwargs["Si"])
        Mntm -= self.stepSize*grad/2.0
        new_draw = (new_draw+self.stepSize * Mntm).clone().detach().requires_grad_(True)
        Mntm -= self.stepSize*energy_gradient_function(kwargs["decoder"],new_draw,Mntm,kwargs["NSi"],kwargs["Si"])/2.0
        Mntm *= -1
        current_U = energy_function(kwargs["decoder"],new_draw,Mntm,kwargs["NSi"],kwargs["Si"])
        current_K= torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
        Z = torch.rand(self.batch)
        criterion = torch.min(self.one,torch.exp(prev_U+prev_K-current_U-current_K))
        draw_final = new_draw.clone().detach()
        draw_final[Z>=criterion] = (previousSample.clone().detach())[Z>=criterion]

        return draw_final,grad

class Leapfrog_Regular(nn.Module):
    def __init__(self,numSteps,stepSize,**kwargs):
        super(Leapfrog_Regular, self).__init__()
        self.numSteps = numSteps 
        self.stepSize = stepSize
        self.kwargs = kwargs
        #self.one = torch.Tensor([1.0])
        self.one = Variable(torch.tensor(1.0).cuda())

    def forward(self,previousSample,energy_function,numSteps=1,stepSize=0.01):
        new_draw = previousSample.clone().detach().requires_grad_(True)
        Mntm = torch.randn_like(new_draw)
        prev_U,gradient = energy_function(new_draw)
        #prev_K = torch.mm(mntm[0,:],mntm[0,:])/2.0
        prev_K = torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
        Mntm -= stepSize*gradient/2.0
        new_draw = (new_draw+stepSize * Mntm).clone().detach().requires_grad_(True)
        current_U,gradient = energy_function(new_draw)
        current_K= torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
        Z = torch.rand(1).cuda()
        criterion = torch.min(self.one,torch.exp(prev_U+prev_K-current_U-current_K))
        acceptance = (Z<criterion).float().mean().item()
        for _ in range(1,self.numSteps):
            Mntm -= stepSize*gradient
            new_draw = (new_draw+stepSize * Mntm).clone().detach().requires_grad_(True)
            current_U,gradient = energy_function(new_draw)
            current_K= torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
            Z = torch.rand(1).cuda()
            criterion = torch.min(self.one,torch.exp(prev_U+prev_K-current_U-current_K))
            acceptance = (Z<criterion).float().mean().item()
        Mntm -= stepSize*gradient/2.0
        Mntm *= -1
        current_U,gradient = energy_function(new_draw)

        current_K= torch.sum(torch.mul(Mntm,Mntm)*0.5,1)
        Z = torch.rand(1).cuda()
        loghamiltonian = prev_U+prev_K-current_U-current_K
        hamiltonian = torch.exp(loghamiltonian)
        criterion = torch.min(self.one,torch.exp(prev_U+prev_K-current_U-current_K))
        draw_final = new_draw.clone().detach()
        draw_final[Z>=criterion] = (previousSample.clone().detach())[Z>=criterion]
        acceptance = (Z<criterion).float().mean().item()

        U_final = current_U.clone().detach()
        #U_final[Z>=criterion] = prev_U[Z>=criterion]

        return draw_final,U_final,acceptance
