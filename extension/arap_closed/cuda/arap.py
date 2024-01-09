import torch
from .lib import arap_cuda
#import sys
#sys.path.append("../../")
class ClosedArapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2,neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight):
        batchsize, n, _ = xyz1.size()
        _, _, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        neighborList = neighborList.contiguous()
        numNeighbors = numNeighbors.contiguous()
        accnumNeighbors = accnumNeighbors.contiguous()
        weightMatrix = weightMatrix.contiguous()
        rotations = rotations.cuda().contiguous()
        rhs = torch.zeros(n,3).cuda().contiguous()
        arap_cuda.forward(xyz1, xyz2, neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,rhs)
        return rhs 

class ClosedArap(torch.nn.Module):
    def forward(self, xyz1, xyz2,neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight):
        return ClosedArapFunction.apply(xyz1, xyz2,neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight)
