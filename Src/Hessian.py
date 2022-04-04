import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import torch.nn as nn
import random
from torch.autograd import Variable
import time
import numpy as np

def sigmoid(x):
    return 1.0/(1.0+torch.exp(-x))

class Hessian(nn.Module):
    def __init__(self,numSteps,stepSize,**kwargs):
        super(Hessian, self).__init__()
        self.numSteps = numSteps 
        self.stepSize = stepSize
        self.kwargs = kwargs
        #self.one = torch.Tensor([1.0])
        self.one = Variable(torch.tensor(1.0).cuda())

    def forward(self,previousSample,energy_function,numSteps=1,stepSize=0.01,k=5,**kwargs):
        new_draw = previousSample.clone().detach().requires_grad_(True)
        prev_U,gradient,hess = energy_function(kwargs["decoder"],new_draw)

        norm_gradient = gradient/torch.norm(gradient)
        num_components = k
        for _ in range(1):
            Mntm = torch.randn_like(new_draw)
            w = Mntm - torch.mm(Mntm,torch.unsqueeze(norm_gradient,-1))*norm_gradient
            #new_draw = (new_draw+0.1 * w).clone().detach().requires_grad_(True)
            #prev_U,gradient,hess = energy_function(kwargs["decoder"],new_draw)
            #w = Mntm
            '''
            try:
                U,S,V = torch.svd(hess)
            except:
                try:
                    U,S,V = torch.svd(hess + 1e-6*hess.mean()*(torch.rand(hess.size()).cuda()))
                except:
                    new_draw = (new_draw+stepSize * w).clone().detach().requires_grad_(True)
                    current_U,gradient,hess = energy_function(kwargs["decoder"],new_draw)
                    draw_final = new_draw.clone().detach()
                    return draw_final,current_U.clone().detach()
            '''
            U,S,V = torch.linalg.svd(hess)
            #print(S)
            #S = eigen_decomposition.eigenvalues
            #S,indices = torch.sort(S,descending=True)
            #V = eigen_decomposition.eigenvectors[:,indices]
            #V = eigen_decomposition.eigenvectors
            #V = V.float().cuda()
            #print(indices)

            #V = V.cuda()
            '''
            w_pca = torch.zeros_like(w)
            for componentNumber in range(k):
                component = V[:,componentNumber]
                w_pca += torch.sum(w*component)*component
            w -= w_pca
            '''
            #subS = S[S>=1e-2]
            #print(S)
            #subS = S[S>=1e-6]
            #scaled_S = S/torch.min(S)
            subS = S[S>=1e-3]
            #print(S.size(),V.size())
            #if num_components==0:
            #    num_components = k
            num_components= 1024 #subS.size()[0]
            #print(num_components)
            Vk = V[:,:num_components]
            #w_pca = torch.unsqueeze(torch.sum(torch.mm(w,Vk) * Vk,1),0)
            w_pca = torch.zeros(w.size()).float().cuda()
            for compIndex in range(num_components):
                #stepSize = 1/(S[compIndex]**0.1)
                stepSize = S[compIndex]**0.5
                #print(stepSize,S[compIndex])
                #stepSize = min(0.1,stepSize)
                #w_pca += stepSize*(torch.mm(w,torch.unsqueeze(Vk[:,compIndex],-1))*Vk[:,compIndex])
                print(2*0.0001/S[compIndex])
                w_pca += (2*0.0001/S[compIndex])*Vk[:,compIndex]
                #new_draw = (new_draw+stepSize * w_pca).clone().detach().requires_grad_(True)
            #w -= w_pca
            #new_draw = (new_draw+0.1*w).clone().detach().requires_grad_(True)
            new_draw = (new_draw+w_pca).clone().detach().requires_grad_(True)
            current_U,gradient,hess = energy_function(kwargs["decoder"],new_draw)
            #print("Energy:",current_U)
            norm_gradient = gradient/torch.norm(gradient)
        draw_final = new_draw.clone().detach()


        return draw_final,current_U.clone().detach()
