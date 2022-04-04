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

def check_symmetric(a):
    return torch.allclose(a, a.T)

class Hessian(nn.Module):
    def __init__(self,numSteps,stepSize,**kwargs):
        super(Hessian, self).__init__()
        self.numSteps = numSteps 
        self.stepSize = stepSize
        self.kwargs = kwargs
        #self.one = torch.Tensor([1.0])
        self.one = Variable(torch.tensor(1.0).cuda())
        self.t = -1.0

    def forward(self,previousSample,energy_function,numSteps=1,stepSize=0.01,k=5,**kwargs):
        new_draw = previousSample.clone().detach().requires_grad_(True)
        #prev_U,gradient,hess = energy_function(kwargs["decoder"],new_draw)
        energy,hess = energy_function(kwargs["decoder"],new_draw)

        if isinstance(energy,str):
            if energy=="nan":
                return "eig failed"
        #print(check_symmetric(hess))
        #norm_gradient = gradient/torch.norm(gradient)

        #norm_gradient = gradient/torch.norm(gradient)
        self.t += 0.1
        print(self.t)
        for _ in range(1):
            #U,S,V = torch.linalg.svd(hess)
            #Mntm = torch.randn_like(new_draw)
            #w = Mntm - torch.mm(Mntm,torch.unsqueeze(norm_gradient,-1))*norm_gradient

            decomposition = torch.linalg.eigh(hess)
            S = decomposition.eigenvalues
            V = decomposition.eigenvectors
            S,indices = torch.sort(S,descending=False)
            V = V[:,indices]
            num_components = 1
            Vk = V[:,:num_components]
            Sk = S[:num_components]
            beta = torch.randn(num_components)
            beta = beta/torch.norm(beta)
            direction = torch.zeros(new_draw.size()).float().cuda()

            for compIndex in range(num_components):
                #if Sk[compIndex]>0:
                direction += self.t*Vk[:,compIndex]
                #else:
                #    direction += beta[compIndex]*(-1*Vk[:,compIndex])

            #norm_direction = torch.norm(direction)
            #direction = direction/norm_direction

            '''
            delta = 1e-3
            betasq_lambda = 0.0
            for compIndex in range(num_components):
                betasq_lambda += ((beta[compIndex]/norm_direction)**2 * torch.abs(Sk[compIndex]))
               
            stepSize = torch.sqrt((2.0*delta)/betasq_lambda)
            stepSize = max(0.2,min(0.7,stepSize))
            #print(stepSize)
            '''
            stepSize = 1.0
            #stepSize = 0.1

            new_draw = (new_draw+stepSize*direction).clone().detach().requires_grad_(True)
            #new_draw = direction.clone().detach().requires_grad_(True)
            #current_U,gradient,hess = energy_function(kwargs["decoder"],new_draw)
            #print("Energy:",current_U)
            #norm_gradient = gradient/torch.norm(gradient)
        draw_final = new_draw.clone().detach()

        return draw_final
