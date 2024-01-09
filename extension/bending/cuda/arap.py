import torch
from .lib import arap_cuda
#import sys
#sys.path.append("../../")

class ArapRotationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2,neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight):
        batchsize, n, _ = xyz1.size()
        _, _, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        updated_rotations = torch.zeros(batchsize, n,9).float().cuda()
        neighborList = neighborList.contiguous()
        numNeighbors = numNeighbors.contiguous()
        accnumNeighbors = accnumNeighbors.contiguous()
        weightMatrix = weightMatrix.contiguous()
        rotations = rotations.cuda().contiguous()
        updated_rotations = updated_rotations.cuda().contiguous()
        arap_cuda.forward(xyz1, xyz2, neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,updated_rotations)
        return updated_rotations

class Bending(torch.nn.Module):
    def forward(self, xyz1, xyz2,neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight):
        return ArapRotationFunction.apply(xyz1, xyz2,neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight)
