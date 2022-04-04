from __future__ import print_function
import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from pointnet import PointNetEncoder as FullPointNetEncoder
from Atlasnet import AtlasnetDecoder
from extension.grad_arap.cuda.arap import ArapGrad
from extension.arap.cuda.arap import Arap
#from extension.grad_isometric.cuda.isometric import IsometricGrad
import csv

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact


    return c.squeeze()

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.norm = F.normalize

    def forward(self, x):
        x = self.norm(x,p=2,dim=1)
        return x

class ArapGradientLayer(nn.Module):
    def __init__(self,rate):
        super(ArapGradientLayer, self).__init__()
        self.rate = rate
        self.arapgrad = ArapGrad()

    def forward(self,xyz,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,arapWeight):
        energies,gradients = self.arapgrad(xyz,reconstruction,neighborsMatrix,numNeighbors,weightMatrix,arapWeight)
        #reconstruction -= self.rate*gradients

        return reconstruction

class IsometricProject(nn.Module):
    def __init__(self,rate,numsteps):
        super(IsometricProject, self).__init__()
        self.rate = rate
        self.numsteps = numsteps
        self.isometricgrad = IsometricGrad()

    def forward(self,xyz,reconstruction,neighborsMatrix,numNeighbors,isometricWeight):
        beta1 = 0.9 ; beta2 = 0.999
        m = torch.zeros(xyz.size()).float().cuda()
        v = torch.zeros(xyz.size()).float().cuda()
        for i in range(self.numsteps):
            gradients = self.isometricgrad(xyz,reconstruction,neighborsMatrix,numNeighbors,isometricWeight)
            m = beta1*m + (1.0-beta1)*gradients
            v = beta2*v + (1.0-beta2)*torch.pow(gradients,2)
            mh = m/(1-beta1**(i+1))
            vh = v/(1-beta2**(i+1))
            reconstruction -= self.rate*(mh/(torch.sqrt(vh)+1e-9))

        return reconstruction

class ArapProject(nn.Module):
    def __init__(self,rate,numsteps):
        super(ArapProject, self).__init__()
        self.rate = rate
        self.numsteps = numsteps
        self.arapgrad = ArapGrad()
        self.arap = Arap()

    def forward(self,xyz,reconstruction,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,arapWeight):
        beta1 = 0.9 ; beta2 = 0.999
        m = torch.zeros(xyz.size()).float().cuda()
        v = torch.zeros(xyz.size()).float().cuda()
        for i in range(self.numsteps):
            _,rotations = self.arap(xyz,reconstruction,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,arapWeight)
            gradients = self.arapgrad(xyz,reconstruction,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight)
            m = beta1*m + (1.0-beta1)*gradients
            v = beta2*v + (1.0-beta2)*torch.pow(gradients,2)
            mh = m/(1-beta1**(i+1))
            vh = v/(1-beta2**(i+1))
            reconstruction -= self.rate*(mh/(torch.sqrt(vh)+1e-9))

        return reconstruction

    
class PointNetCorr(nn.Module):
    def __init__(self, npoint=6890, nlatent=32,n_dims=3):
        """Encoder"""

        super(PointNetCorr, self).__init__()
        self.conv1 = torch.nn.Conv1d(n_dims, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1024, 2048, 1)
        self.conv5 = torch.nn.Conv1d(2048, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)
        self.lin3 = nn.Linear(nlatent, nlatent)

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(2048)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)
        self.l2 = Normalize()

        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = self.lin1(x)
        return self.l2(x) #.squeeze(2)

class PointNetFeat(nn.Module):
    def __init__(self, npoint=6890, nlatent=32,n_dims=3):
        """Encoder"""

        super(PointNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(n_dims, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)
        self.lin3 = nn.Linear(nlatent, nlatent)

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)
        self.l2 = Normalize()

        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        #x = self.lin1(x).unsqueeze(-1)
        mu = self.lin2(x.squeeze(2)) #.unsqueeze(-1)
        std = F.relu(self.lin3(x.squeeze(2))) #.unsqueeze(-1)))
        eps = torch.randn_like(std)
        x = mu + std * eps

        x = self.l2(x)
        return x #.squeeze(2)

'''
def cov(data):
    dmean = torch.mean(data, dim=0)
    centered_data = data - dmean.expand_as(data)
    return torch.mm(centered_data.transpose(0,1), centered_data)
'''

class NormalReg(nn.Module):

    def __init__(self,regWeight=1.0):
        super(NormalReg, self).__init__()

        self.regWeight = regWeight

    def forward(self, x):
        mean = torch.mean(x, dim=0).pow(2)
        covar = cov(x)
        cov_loss = torch.mean(
                (Variable(torch.eye(covar.size()[0]).cuda())-covar)
                .pow(2))

        return self.regWeight*(torch.mean(mean) + cov_loss)

class Position(nn.Module):
    def __init__(self, init):
        super(Position, self).__init__()
        self.proposed = init.clone().detach().requires_grad_(True)
        self.proposed = torch.nn.Parameter(self.proposed)

class CodePosition(nn.Module):
    def __init__(self, init):
        super(CodePosition, self).__init__()
        self.proposed = init.clone().detach().requires_grad_(True)
        self.proposed = torch.nn.Parameter(self.proposed)

class Latents(nn.Module):
    def __init__(self, numRows,numCols):
        super(Latents, self).__init__()
    
        #self.mu = torch.zeros((numRows,numCols),device="cuda",dtype=torch.float,requires_grad=True)
        #self.std = torch.ones((numRows,numCols),device="cuda",dtype=torch.float,requires_grad=True)
        #self.mu = torch.nn.Parameter(self.mu)
        #self.std = torch.nn.Parameter(self.std)
        self.code = torch.randn((numRows,numCols),device="cuda",dtype=torch.float,requires_grad=True)
        self.code = torch.nn.Parameter(self.code)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
    
    def forward(self):
        return self.mu,self.std

class SingleLayerClassifier(nn.Module):
    def __init__(self, input_dim,num_classes):
        super(SingleLayerClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim,num_classes)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))

        return x

class PointNetEncoderFC(nn.Module):
    def __init__(self, num_points = 2500,bottleneck=1024,num_dims=3):
        super(PointNetEncoderFC, self).__init__()
        self.num_points = num_points
        self.bottleneck = bottleneck 
        self.num_dims  = num_dims
        self.fc1 = torch.nn.Linear(num_points*num_dims,1024)
        self.fc2 = torch.nn.Linear(1024,2048)
        self.fc3 = torch.nn.Linear(2048,4096)
        self.fc4 = torch.nn.Linear(4096,2048)
        self.fc5 = torch.nn.Linear(2048,1024)
        self.fc6 = torch.nn.Linear(1024,bottleneck)
        #self.fcmu = nn.Linear(1024,bottleneck)
        self.fcvar = torch.nn.Linear(1024,bottleneck)
        self.l2 = Normalize()
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(2048)

    def forward(self, x):
        x = x.contiguous().view(-1,self.num_points*self.num_dims)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.bn2(self.fc4(x)))
        x = F.relu(self.bn1(self.fc5(x)))
        #x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        x = self.fc6(x)
        #mu = self.l2(mu)
        #mu = torch.tanh(self.fcmu(x))
        #std = torch.sigmoid(self.fcvar(x))

        #eps = torch.randn_like(std)
        #x = mu + std * eps

        #x = mu + eps*var

        return x #,mu,std

class PointNetDecoderFC(nn.Module):
    def __init__(self, num_points = 2500,bottleneck=1024,num_dims=3):
        super(PointNetDecoderFC, self).__init__()
        self.num_points = num_points
        self.num_dims = 3 
        self.bottleneck = bottleneck 
        self.fc1 = torch.nn.Linear(bottleneck,512)
        self.fc2 = torch.nn.Linear(512,1024)
        self.fc6 = torch.nn.Linear(1024,3*num_points)

        '''
        self.fc1 = torch.nn.Linear(bottleneck,1024)
        self.fc2 = torch.nn.Linear(1024,2048)
        self.fc3 = torch.nn.Linear(2048,4096)
        self.fc4 = torch.nn.Linear(4096,2048)
        self.fc5 = torch.nn.Linear(2048,1024)
        self.fc6 = torch.nn.Linear(1024,3*num_points)
        '''

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        x = self.fc6(x)
        reconstruction = x.view(x.shape[0],self.num_points,self.num_dims)
        return reconstruction

'''
class PointNetEncoder1D(nn.Module):
    def __init__(self, num_points = 2500,bottleneck=1024,num_dims=3):
        super(PointNetEncoder1D, self).__init__()
        self.num_points = num_points
        self.bottleneck = bottleneck 
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, bottleneck, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(bottleneck)
        self.norm = Normalize()

    def forward(self, x):
        batchsize = x.size()[0]
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = (self.conv4(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #x = torch.mean(x, 2)
        #print(x.size())
        x = x.view(-1, self.bottleneck)
        #x = self.norm(x)

        return x
'''

class PointNetEncoder1D(nn.Module):
    def __init__(self, num_points = 2500,bottleneck=1024,num_dims=3):
        super(PointNetEncoder1D, self).__init__()
        self.num_points = num_points
        self.bottleneck = bottleneck 
        self.conv1 = torch.nn.Conv1d(3, 6, 10)
        self.max1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(6, 6, 10)
        self.max2 = torch.nn.MaxPool1d(2)
        self.conv3 = torch.nn.Conv1d(6, 12, 10)
        self.max3 = torch.nn.MaxPool1d(2)
        self.conv4 = torch.nn.Conv1d(12, 12, 10)
        self.max4 = torch.nn.MaxPool1d(2)
        self.conv5 = torch.nn.Conv1d(12, 16, 10)
        self.max5 = torch.nn.MaxPool1d(2)
        self.conv6 = torch.nn.Conv1d(16, 16, 10)
        self.max6 = torch.nn.MaxPool1d(2)
        self.lin1 = torch.nn.Linear(1440,bottleneck)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(bottleneck)
        self.norm = Normalize()

    def forward(self, x):
        batchsize = x.size()[0]
        x = (F.relu(self.conv1(x)))
        x = self.max1(x)
        x = (F.relu(self.conv2(x)))
        x = self.max2(x)
        x = (F.relu(self.conv3(x)))
        x = self.max3(x)
        x = x.view(x.size()[0],-1)
        x = self.lin1(x)
        #x = (F.relu(self.conv4(x)))
        #x = self.max4(x)
        #x = (F.relu(self.conv5(x)))
        #x = self.max5(x)
        #x = (F.relu(self.conv6(x)))
        #x = self.max6(x)
        '''
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = (self.conv4(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #x = torch.mean(x, 2)
        #print(x.size())
        x = x.view(-1, self.bottleneck)
        #x = self.norm(x)
        '''

        return x

class PointNetEncoder(nn.Module):
    def __init__(self,num_points=1024,num_dims=3,bottleneck=1024):
        super(PointNetEncoder,self).__init__()
        self.conv1 = nn.Conv2d(1,512,(4,1))
        self.conv2 = nn.Conv2d(512,1024,(3,1))
        self.conv3 = nn.Conv2d(1024,2048,(3,1))
        self.conv4 = nn.Conv2d(2048,bottleneck,(3,num_dims))
        self.conv5 = nn.Conv2d(512,bottleneck,(1,1))
        self.bn1 = nn.BatchNorm2d(128)
        '''
        '''
        self.bn1 = nn.BatchNorm2d(32,track_running_stats=False,momentum=0.01)
        self.bn2 = nn.BatchNorm2d(64,track_running_stats=False,momentum=0.01)
        self.bn3 = nn.BatchNorm2d(32,track_running_stats=False,momentum=0.01)
        self.l2 = Normalize()
        self.bottleneck = bottleneck

        self.fcmu = nn.Linear(1024,bottleneck)
        self.fcvar = nn.Linear(1024,bottleneck)

    def forward(self,pts):
        x = (F.relu(self.conv1(pts)))
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = self.conv4(x)
        x,_ = torch.max(x,2,keepdim=True)
        #x = torch.mean(x,2,keepdim=True)
        x = x.view(-1,self.bottleneck)
        #x = self.l2(x)
        #mu = torch.tanh(self.fcmu(x))
        #var = torch.sigmoid(self.fcvar(x))

        #eps = 0.001*torch.randn_like(var)
        #x = mu + torch.exp(var/ 2) * eps

        #x = mu + eps*var
        return x

class PointNetDecoder(nn.Module):
    def __init__(self,num_points=1024,num_dims=6,bottleneck=1024,catSize=1):
        super(PointNetDecoder,self).__init__()
        self.fc1 = nn.Linear(bottleneck,bottleneck)
        self.bn1 = nn.BatchNorm1d(num_points)
        self.fc2 = nn.Linear(bottleneck,num_points*3)
        self.bottleneck = bottleneck

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size()[0],-1,3)
        return x

'''
class PointNetDecoderConv(nn.Module):
    def __init__(self,num_points=1024,num_dims=6,bottleneck=1024,catSize=1):
        super(PointNetDecoderConv,self).__init__()
        self.deconv1 = nn.ConvTranspose1d(bottleneck,512,2,1)
        self.deconv2 = nn.ConvTranspose1d(512,256,2,2)
        self.deconv3 = nn.ConvTranspose1d(256,128,2,2)
        self.deconv4 = nn.ConvTranspose1d(128,64,2,2)
        self.deconv5 = nn.ConvTranspose1d(64,32,2,2)
        self.deconv6 = nn.ConvTranspose1d(32,16,2,2)
        self.deconv7 = nn.ConvTranspose1d(16,8,2,2)
        self.deconv8 = nn.ConvTranspose1d(8,4,2,2)
        self.deconv9 = nn.ConvTranspose1d(4,4,2,2)
        self.deconv10 = nn.ConvTranspose1d(4,3,2,2)

    def forward(self,x):
        x = x.view(x.size()[0],-1,1)
        x =(F.relu(self.deconv1(x)))
        x =(F.relu(self.deconv2(x)))
        x =(F.relu(self.deconv3(x)))
        x =(F.relu(self.deconv4(x)))
        x =(F.relu(self.deconv5(x)))
        x =(F.relu(self.deconv6(x)))
        x =(F.relu(self.deconv7(x)))
        x =(F.relu(self.deconv8(x)))
        x =(F.relu(self.deconv9(x)))
        x = (self.deconv10(x))
        x = x.view(x.size()[0],1024,3)
        return x
'''
class PointNetDecoderConvShallow2(nn.Module):
    def __init__(self,num_points=1024,num_dims=6,bottleneck=1024,catSize=1):
        super(PointNetDecoderConvShallow2,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(bottleneck,2048,(1,1),(1,1))
        self.deconv2 = nn.ConvTranspose2d(2048,1024,(2,1),(1,1))
        self.deconv3 = nn.ConvTranspose2d(1024,512,(1,1),(1,1))
        self.deconv4 = nn.ConvTranspose2d(512,2*num_dims,(3,2),(1,1))

        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(3)
        self.num_points = num_points
        self.num_dims = num_dims

    def forward(self,x):
        x = x.view(x.size()[0],-1,1,1)
        x = (F.relu(self.deconv1(x)))
        x = (F.relu(self.deconv2(x)))
        x = (F.relu(self.deconv3(x)))
        x = self.deconv4(x)
        x = x.view(x.size()[0],self.num_points,self.num_dims)
        return x

class PointNetDecoderConvShallow(nn.Module):
    def __init__(self,num_points=1024,num_dims=6,bottleneck=1024,catSize=1):
        super(PointNetDecoderConvShallow,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(bottleneck,2048,(2,2),(1,1))
        self.conv1 = nn.Conv2d(2048,2048,(1,1))
        self.deconv2 = nn.ConvTranspose2d(2048,1024,(2,2),(1,1))
        self.conv2 = nn.Conv2d(1024,1024,(2,2))
        self.deconv3 = nn.ConvTranspose2d(1024,512,(3,3),(1,1))
        self.conv3 = nn.Conv2d(512,num_dims,(1,1))

        self.bn1 = nn.BatchNorm2d(32,track_running_stats=False,momentum=0.01)
        self.bn2 = nn.BatchNorm2d(64,track_running_stats=False,momentum=0.01)
        self.num_points = num_points
        self.num_dims = num_dims

    def forward(self,x):
        x = x.view(x.size()[0],-1,1,1)
        x = (F.relu(self.deconv1(x)))
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.deconv2(x)))
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.deconv3(x)))
        x = (self.conv3(x))
        x = x.view(x.size()[0],self.num_points,self.num_dims)
        return x

class PointNetDecoderConv(nn.Module):
    def __init__(self,num_points=1024,num_dims=6,bottleneck=1024,catSize=1):
        super(PointNetDecoderConv,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(bottleneck,512,(2,2),(1,1))
        self.deconv2 = nn.ConvTranspose2d(512,256,(2,2),(1,1))
        self.deconv3 = nn.ConvTranspose2d(256,128,(3,2),(2,2))
        self.deconv4 = nn.ConvTranspose2d(128,64,(3,5),(2,3))
        self.deconv5 = nn.ConvTranspose2d(64,num_dims,(1,1),(1,1))
        self.deconv6 = nn.ConvTranspose2d(32,num_dims,(5,5),(5,5))

        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(3)
        self.num_points = num_points
        self.num_dims = num_dims

    def forward(self,x):
        x = x.view(x.size()[0],-1,1,1)
        x = (F.relu(self.deconv1(x)))
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.deconv2(x)))
        x = (F.relu(self.deconv3(x)))
        x = (F.relu(self.deconv4(x)))
        x = ((self.deconv5(x)))
        #x = self.deconv6(x)
        #x = x.view(x.size()[0],-1)[:,:self.num_points]
        #print(x.size())
        x = x.view(x.size()[0],self.num_points,self.num_dims)
        return x

class DistributionNetConv(nn.Module):
    def __init__(self,bottleneck=1024,num_dims=6,catSize=1):
        super(DistributionNetConv,self).__init__()
        self.fc1Nmu = nn.Linear(bottleneck,bottleneck)
        self.fc1Nvar = nn.Linear(bottleneck,bottleneck)

        self.conv1 = nn.Conv2d(1,128,(1,num_dims))
        self.conv2 = nn.Conv2d(128,256,(1,1))
        self.conv3 = nn.Conv2d(256,512,(1,1))
        self.conv4 = nn.Conv2d(512,bottleneck,(1,1))
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.l2 = Normalize()
        self.bottleneck = bottleneck

    def forward(self,x):
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = self.conv4(x)
        x,_ = torch.max(x,2,keepdim=True)
        x = x.view(-1,self.bottleneck)
        #xMu = torch.tanh(self.fc1Nmu(x))
        #xStd = torch.sigmoid(self.fc1Nvar(x))
        xMu = self.fc1Nmu(x)
        xLogvar = self.fc1Nvar(x)
        xStd = torch.exp(0.5*xLogvar)
        eps = torch.randn_like(xStd)
        sample = xMu + eps*xStd
        return xMu,xStd,sample

class DistributionNet(nn.Module):
    def __init__(self,bottleneck=1024,num_dims=6,catSize=1):
        super(DistributionNet,self).__init__()
        self.fc1 = nn.Linear(bottleneck+catSize,bottleneck)
        #self.bn1 = nn.BatchNorm1d(bottleneck)
        self.fc1Nmu = nn.Linear(bottleneck,bottleneck)
        self.fc1Nvar = nn.Linear(bottleneck,bottleneck)

        self.conv1 = nn.Conv2d(1,128,(1,num_dims))
        self.conv2 = nn.Conv2d(128,256,(1,1))
        self.conv3 = nn.Conv2d(256,512,(1,1))
        self.conv4 = nn.Conv2d(512,bottleneck,(1,1))
        '''
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        '''
        self.l2 = Normalize()
        self.bottleneck = bottleneck

    def forward(self,x):
        '''
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = (F.relu(self.conv3(x)))
        x = self.conv4(x)
        x,_ = torch.max(x,2,keepdim=True)
        x = x.view(-1,self.bottleneck)
        '''
        #x = self.l2(x)
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        #xMu = torch.tanh(self.fc1Nmu(x))
        #xStd = torch.sigmoid(self.fc1Nvar(x))
        xMu = self.fc1Nmu(x)
        xLogvar = self.fc1Nvar(x)
        xStd = torch.exp(0.5*xLogvar)
        eps = torch.randn_like(xStd)
        sample = xMu + eps*xStd
        return xMu,xStd,sample

class PointNetAutoEncoder(nn.Module):
    def __init__(self,num_points=1024,num_dims=6,bottleneck=1024,catSize=1,numClasses=2):
        super(PointNetAutoEncoder,self).__init__()
        self.encoder = PointNetEncoder(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck)
        #self.encoder = Atlasnet_Encoder(num_points=num_points,bottleneck_size=bottleneck)
        self.encoder2 = PointNetEncoder(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck)
        self.encoder3 = PointNetEncoder(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck)
        self.distnet = DistributionNet(bottleneck=bottleneck,catSize=catSize,num_dims=6)
        self.decoder = PointNetDecoderConv(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck,catSize=0)
        #self.decoder = Atlasnet_Decoder(num_points=num_points,bottleneck_size=bottleneck,catSize=catSize,nb_primitives=8)
        self.classLayer = nn.Linear(bottleneck,numClasses)
        self.sampleClassLayer = nn.Linear(bottleneck,numClasses)
        self.catSize = catSize
        #self.decoder = Atlasnet_Decoder(num_points=num_points,bottleneck_size=bottleneck,nb_primitives=25)

    def forward(self,pts,condition,otherPts):
        code = self.encoder(pts)
        #jcode2 = self.encoder2(pts)
        #code3 = self.encoder3(pts)
        classPred = torch.sigmoid(self.classLayer(code))
        #otherCode = self.encoder(otherPts)
        #ptsCond = torch.cat((pts,condition),-1)
        #otherPtsCond = torch.cat((otherPts,otherCond),-1)
        #decCondition = otherCond[:,:,:50,:]
        #decCondition = decCondition.view(decCondition.size()[0],-1)
        #code = self.encoder(ptsCond)
        #condition = torch.squeeze(condition)
        condition = condition.view(condition.size()[0],-1) #[:,:self.catSize]
        conditionCode = torch.cat((code,condition),1)
        #otherConditionCode = torch.cat((otherCode,condition),1)
        mu,std,sample = self.distnet(conditionCode)
        #_,_,sample = self.distnet(otherConditionCode)
        #_,_,sample = self.distnet(otherPtsCond)
        codeSample = torch.cat((code,condition,sample),1)
        reconstruction = self.decoder(codeSample)

        return mu,std,reconstruction,code,sample,classPred

class PointNetAutoEncoderConv(nn.Module):
    def __init__(self,num_points=1024,num_dims=6,bottleneck=1024,catSize=1,numClasses=2):
        super(PointNetAutoEncoderConv,self).__init__()
        self.encoder = PointNetEncoder(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck)
        self.encoder2 = PointNetEncoder(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck)
        self.encoder3 = PointNetEncoder(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck)
        self.distnet = DistributionNetConv(bottleneck=bottleneck,catSize=catSize,num_dims=6)
        self.decoder = PointNetDecoderConv(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck,catSize=catSize)
        self.classLayer = nn.Linear(bottleneck,numClasses)
        self.sampleClassLayer = nn.Linear(bottleneck,numClasses)
        self.catSize = catSize

    def forward(self,pts,condition,otherPts):
        code = self.encoder(pts)
        #code2 = self.encoder2(pts)
        #code3 = self.encoder3(pts)
        classPred = torch.sigmoid(self.classLayer(code))
        ptsCond = torch.cat((pts,condition),-1)
        #otherPtsCond = torch.cat((otherPts,otherCond),-1)
        #code = self.encoder(ptsCond)
        #condition = torch.squeeze(condition)
        #otherConditionCode = torch.cat((otherCode,condition),1)
        mu,std,sample = self.distnet(ptsCond)
        decCondition = condition.view(condition.size()[0],-1)
        decCondition = decCondition[:,:self.catSize]
        codeSample = torch.cat((code,sample,decCondition),1)
        reconstruction = self.decoder(codeSample)

        return mu,std,reconstruction,code,sample,classPred

class VertexNetAutoEncoder2(nn.Module):
    def __init__(self,num_points=1024,num_dims=6,bottleneck=1024,catSize=1,numClasses=2):
        super(VertexNetAutoEncoder2,self).__init__()
        self.encoder = PointNetEncoder(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck)
        #self.encoder = Atlasnet_Encoder(num_points=num_points,bottleneck=bottleneck)
        self.regularizer = NormalReg()
        self.distnet = DistributionNet(bottleneck=bottleneck,catSize=0)
        self.classLayer = nn.Linear(bottleneck,numClasses)
        #self.decoder = PointNetDecoderConvShallow2(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck,catSize=0)
        self.decoder = PointNetDecoderConvShallow(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck,catSize=0)
        self.num_points = num_points

    def forward(self,pts):
        code = self.encoder(pts)
        mu,std,sample = self.distnet(code)
        reconstruction = self.decoder(sample)
        #reconstruction = reconstruction.view(reconstruction.size()[0],self.num_points,-1)
        return mu,std,sample,reconstruction #.permute(0,2,1)

class VertexNetAutoEncoder(nn.Module):
    def __init__(self,num_points=1024,num_dims=6,bottleneck=1024,catSize=1,numClasses=2):
        super(VertexNetAutoEncoder,self).__init__()
        self.encoder = PointNetEncoder(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck)
        #self.encoder = Atlasnet_Encoder(num_points=num_points,bottleneck=bottleneck)
        self.regularizer = NormalReg()
        self.distnet = DistributionNet(bottleneck=bottleneck,catSize=0)
        self.classLayer = nn.Linear(bottleneck,numClasses)
        self.decoder = PointNetDecoderConvShallow2(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck,catSize=0)
        #self.decoder = PointNetDecoderConv(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck,catSize=0)
        #self.decoder = AtlasnetDecoder(num_points=num_points,nb_primitives=4,bottleneck=bottleneck)
        self.num_points = num_points

    def forward(self,pts):
        code = self.encoder(pts)
        mu,std,sample = self.distnet(code)
        reconstruction = self.decoder(sample)
        #reconstruction = reconstruction.view(reconstruction.size()[0],self.num_points,-1)
        return mu,std,sample,reconstruction #.permute(0,2,1)

class VertexNetAutoEncoderFC(nn.Module):
    def __init__(self,num_points=1024,num_dims=3,bottleneck=1024,catSize=1,numClasses=2,rate=0.001,numsteps=1,energy='arap'):
        super(VertexNetAutoEncoderFC,self).__init__()
        self.encoder = PointNetFeat(npoint=num_points,nlatent=bottleneck,n_dims=num_dims)
        #self.encoder = PointNetEncoderFC(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck)
        #self.decoder1 = AtlasnetDecoder(num_points=num_points,nb_primitives=8,bottleneck=bottleneck)
        self.decoder1 = PointNetDecoderFC(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck)
        #if energy=='arap':
        #    self.decoder2 = ArapProject(rate=rate,numsteps=numsteps)
        #    self.decoder2.requires_grad_(False)
        #self.decoder = PointNetDecoderFC(num_points=num_points,num_dims=num_dims,bottleneck=bottleneck)
        self.num_points = num_points

    def forward(self,pts):
        pts = pts.view(pts.size()[0],-1)
        mu,std,sample = self.encoder(pts)
        reconstruction = self.decoder(sample)
        #reconstruction = reconstruction.view(reconstruction.size()[0],self.num_points,-1)
        return mu,std,sample,reconstruction #.permute(0,2,1)

class FCEncoder(nn.Module):
    def __init__(self,bottleneck=4):
        super(FCEncoder,self).__init__()
        self.fc1 = nn.Linear(32,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,bottleneck)
        self.l2 = Normalize() 
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.l2(x)

        return x
    
class FCDecoder(nn.Module):
    def __init__(self,bottleneck=4):
        super(FCDecoder,self).__init__()
        self.fc1 = nn.Linear(bottleneck,8)
        self.fc2 = nn.Linear(8,16)
        self.fc3 = nn.Linear(16,32)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class FCAutoEncoder(nn.Module):
    def __init__(self,bottleneck=1024):
        super(FCAutoEncoder,self).__init__()
        self.encoder = FCEncoder(bottleneck=bottleneck)
        #self.encoder = Atlasnet_Encoder(num_points=num_points,bottleneck=bottleneck)
        self.regularizer = NormalReg()
        self.decoder = FCDecoder(bottleneck=bottleneck)

    def forward(self,pts):
        code = self.encoder(pts)
        reconstruction = self.decoder(code)
        #reconstruction = reconstruction.view(reconstruction.size()[0],self.num_points,-1)
        return reconstruction #.permute(0,2,1)
