import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
sys.path.append("../")
from Dataloader import ScapeObjMesh
import json
from Utils.reader import readPoints
from Networks import Positions
import random
import torch.nn as nn
from Utils import writer,Off,Obj
from extension.carap.cuda.carap import CArap
from extension.arap.cuda.arap import Arap
from extension.asap.cuda.asap import Asap
import numpy as np
import os
import torch.nn as nn
def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight) #,nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.0)
    elif classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight) #,nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.0)
    elif classname.find('ConvTranspose2d')!=-1:
        nn.init.xavier_normal_(m.weight) #,nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.0)


config = sys.argv[1] 
args = json.loads(open(config,'r').read())
if not os.path.exists(args['sampleDir']):
    os.makedirs(args['sampleDir'])

srcShapePath = args['sourceShape']
srcObj = Obj.Obj(srcShapePath)
srcObj.readContents()
srcObj.computeArapMetadata()
srcPoints,_ = srcObj.readPointsAndFaces(normed=True)
srcpts = torch.unsqueeze(torch.from_numpy(srcPoints).float().cuda(),0)
srcpts = srcpts[:,:,:2]

targetShapePath = args['targetShape']
targetObj = Obj.Obj(targetShapePath)
targetObj.readContents()
targetObj.computeArapMetadata()


position = Positions(init=srcpts).cuda()
arap = Arap()
points,faces = targetObj.readPointsAndFaces(normed=True)
numNeighbors,neighborsMatrix,weightMatrix = targetObj.getArapMetadata()
area = targetObj.getArea()
area = np.array([area])

pts = torch.unsqueeze(torch.from_numpy(points).float().cuda(),0)
pts = pts[:,:,:2]
neighborsMatrix = torch.unsqueeze(torch.from_numpy(neighborsMatrix).int().cuda(),0)
weightMatrix = torch.unsqueeze(torch.from_numpy(weightMatrix).float().cuda(),0)
numNeighbors = torch.unsqueeze(torch.from_numpy(numNeighbors).int().cuda(),0)
areas = torch.unsqueeze(torch.from_numpy(area).float().cuda(),0)
alpha = 0.001
weight = Variable(torch.tensor(1.0)).cuda()

epochs = 10000
optimizer = torch.optim.Adam(position.parameters(),lr=1e-2,weight_decay=0)
mse = nn.MSELoss()
position.train()
bestError=1e9
for i in range(epochs):

    optimizer.zero_grad()
    loss = arap(pts,position.proposed,neighborsMatrix,numNeighbors,weightMatrix,weight).mean()
    loss.backward()
    optimizer.step()
    reconstructionErr = mse(pts,position.proposed)
    print(i,loss.item(),reconstructionErr)

    proposedPoints = position.proposed.detach().cpu().numpy()[0]
    z = np.zeros((args['numPoints'],1))
    proposedPoints = np.concatenate([proposedPoints,z],1)
    if loss.item()<bestError:
        objInstance = Obj.Obj("dummy.obj")
        objInstance.setVertices(proposedPoints)
        objInstance.setFaces(faces)
        objInstance.saveAs(os.path.join(args['sampleDir'],"gbman_def7_"+str(i)+".obj"))
