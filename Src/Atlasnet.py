from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class PointGenCon(nn.Module):
    def __init__(self, bottleneck = 2500):
        self.bottleneck = bottleneck
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck,1024, 1)
        self.conv2 = torch.nn.Conv1d(1024,512, 1)
        self.conv3 = torch.nn.Conv1d(512,256, 1)
        self.conv4 = torch.nn.Conv1d(256,3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class AtlasnetDecoder(nn.Module):
    def __init__(self, num_points = 2048, bottleneck= 1024, nb_primitives = 4):
        super(AtlasnetDecoder, self).__init__()
        self.num_points = num_points
        self.bottleneck = bottleneck
        self.nb_primitives = nb_primitives

        self.decoder = nn.ModuleList([PointGenCon(bottleneck = 2 +self.bottleneck) for i in range(0,self.nb_primitives)])

    def forward(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.nb_primitives))
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
