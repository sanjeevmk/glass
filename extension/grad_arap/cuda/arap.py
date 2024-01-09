import torch
from .lib import arap_cuda as grad_arap_cuda

class ArapGradFunction(torch.autograd.Function):
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
        cellEnergies = torch.zeros(batchsize, n,3)
        gradients = torch.zeros(batchsize, n,3)
        cellEnergies = cellEnergies.cuda().contiguous()
        gradients = gradients.cuda().contiguous()
        grad_arap_cuda.forward(xyz1, xyz2, neighborList,numNeighbors,accnumNeighbors,weightMatrix,cellEnergies,rotations,gradients)
        ctx.save_for_backward(xyz1, xyz2, neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight)
        return gradients

    @staticmethod
    def backward(ctx,grad_output):
        xyz1, xyz2, neighborList, numNeighbors, accnumNeighbors, weightMatrix, rotations, arapWeight = ctx.saved_tensors
        gradxyz = torch.zeros_like(grad_output)
        grad_arap_cuda.backward(xyz1, xyz2, neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,gradxyz)
        return None,grad_output*gradxyz,None,None,None,None,None,None


class ArapGrad(torch.nn.Module):
    def forward(self, xyz1, xyz2,neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight):
        return ArapGradFunction.apply(xyz1, xyz2,neighborList,numNeighbors,accnumNeighbors,weightMatrix,rotations,arapWeight)
