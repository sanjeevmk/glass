import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
sys.path.append("../")
from Dataloader import ScapeOffPoints,ScapeOffMesh,ToscaPoints,ScapeObjMesh,ScapeObjMeshRotate,ScapeObjPoints,ScapeObjMeshPairs,ToonObjMesh,FAUSTSimple,FAUSTObj,ScapeObjMeshPairs,Dfaust,ScapeObjMeshFullArap
import json
from Utils.reader import readPoints
from Networks import VertexNetAutoEncoder,NormalReg,CodePosition,Position,VertexNetAutoEncoderFC,SingleLayerClassifier,PointNetFeat,PointNetCorr
import random
import torch.nn as nn
import numpy as np
from Leapfrog import Leapfrog
from nuts.nuts import nuts6
import os
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight) #,nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.0)
    elif classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight) #,nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.0)
    elif classname.find('ConvTranspose2d')!=-1:
        nn.init.xavier_normal_(m.weight) #,nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.0)
