import torch
from .lib import arap_cpu

class ArapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2,neighborList,numNeighbors,weightMatrix):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        neighborList = neighborList.contiguous()
        numNeighbors = numNeighbors.contiguous()
        weightMatrix = weightMatrix.contiguous()

        cellEnergies = torch.zeros(batchsize, n,3).contiguous()
        rotations = torch.zeros(batchsize, n,9).contiguous()

        arap_cpu.forward(xyz1, xyz2, neighborList,numNeighbors,weightMatrix,cellEnergies,rotations)

        ctx.save_for_backward(xyz1, xyz2, neighborList,numNeighbors,weightMatrix,rotations)
        return 1e-4*cellEnergies

    @staticmethod
    def backward(ctx,grad_output):
        xyz1, xyz2, neighborList, numNeighbors, weightMatrix, rotations = ctx.saved_tensors
        gradxyz = grad_output.cpu()
        arap_cpu.backward(xyz1, xyz2, neighborList,numNeighbors,weightMatrix,rotations,gradxyz)
        return None,1e-4*gradxyz,None,None,None


class Arap(torch.nn.Module):
    def forward(self, xyz1, xyz2,neighborList,numNeighbors,weightMatrix):
        return ArapFunction.apply(xyz1, xyz2,neighborList,numNeighbors,weightMatrix)
