import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import torch.nn as nn
import random
from torch.autograd import Variable
import time

class Tangent(nn.Module):
    def __init__(self,numSteps,stepSize,**kwargs):
        super(Tangent, self).__init__()
        self.numSteps = numSteps 
        self.stepSize = stepSize
        self.kwargs = kwargs
        #self.one = torch.Tensor([1.0])
        self.one = Variable(torch.tensor(1.0).cuda())

    def forward(self,previousSample,energy_function,numSteps=1,stepSize=0.01,**kwargs):
        new_draw = previousSample.clone().detach().requires_grad_(True)
        Mntm = torch.randn_like(new_draw)
        prev_U,gradient = energy_function(kwargs["decoder"],new_draw)
        norm_v = torch.norm(Mntm)
        norm_gradient = gradient/torch.norm(gradient)
        for _ in range(self.numSteps):
            w = Mntm - (torch.sum(Mntm*norm_gradient)*norm_gradient)
            #w = w * (norm_v/torch.norm(w))
            new_draw = (new_draw+stepSize * w).clone().detach().requires_grad_(True)
            current_U,gradient = energy_function(kwargs["decoder"],new_draw)
            norm_gradient = gradient/torch.norm(gradient)
        draw_final = new_draw.clone().detach()


        return draw_final,current_U.clone().detach()
