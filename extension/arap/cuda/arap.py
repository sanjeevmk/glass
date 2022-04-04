import torch
from .lib import arap_cuda
#import sys
#sys.path.append("../../")
from extension.grad_arap.cuda.arap import ArapGrad
class ArapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2,neighborList,numNeighbors,accnumNeighbors,weightMatrix,arapWeight):
        batchsize, n, _ = xyz1.size()
        _, _, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        cellEnergies = torch.zeros(batchsize, n,3)
        rotations = torch.zeros(batchsize, n,9)
        neighborList = neighborList.contiguous()
        numNeighbors = numNeighbors.contiguous()
        accnumNeighbors = accnumNeighbors.contiguous()
        weightMatrix = weightMatrix.contiguous()
        cellEnergies = cellEnergies.cuda().contiguous()
        rotations = rotations.cuda().contiguous()
        arap_cuda.forward(xyz1, xyz2, neighborList,numNeighbors,accnumNeighbors,weightMatrix,cellEnergies,rotations)
        ctx.save_for_backward(xyz1, xyz2, neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight)
        return arapWeight*cellEnergies,rotations

    @staticmethod
    def backward(ctx,grad_output,gradrot_output):
        xyz1, xyz2, neighborList, numNeighbors, accnumNeighbors, weightMatrix, rotations, arapWeight = ctx.saved_tensors
        #arap_cuda.backward(xyz1, xyz2, neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,gradxyz)

        #gradxyz = grad_output.clone()
        #gradxyz = gradxyz.cuda().contiguous()
        arapGrad = ArapGrad()
        gradxyz = grad_output * arapGrad(xyz1, xyz2, neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight)
        return None,arapWeight*gradxyz,None,None,None,None,None


class Arap(torch.nn.Module):
    def forward(self, xyz1, xyz2,neighborList,numNeighbors,accnumNeighbors,weightMatrix,arapWeight):
        return ArapFunction.apply(xyz1, xyz2,neighborList,numNeighbors,accnumNeighbors,weightMatrix,arapWeight)
