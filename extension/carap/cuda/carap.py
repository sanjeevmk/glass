import torch
from .lib import carap_cuda
class CArapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2,neighborList,numNeighbors,weightMatrix,alpha,area,carapWeight):
        batchsize, n, _ = xyz1.size()
        _, _, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        neighborList = neighborList.contiguous()
        numNeighbors = numNeighbors.contiguous()
        weightMatrix = weightMatrix.contiguous()
        cellEnergies = torch.zeros(batchsize, n,3)
        rotations = torch.zeros(batchsize, n,9)
        alphas = torch.ones(batchsize,)*alpha
        cellEnergies = cellEnergies.cuda().contiguous()
        rotations = rotations.cuda().contiguous()
        alphas = alphas.cuda().contiguous()
        areas = area.cuda().contiguous()
        carap_cuda.forward(xyz1, xyz2, neighborList,numNeighbors,weightMatrix,alphas,areas,cellEnergies,rotations)
        ctx.save_for_backward(xyz1, xyz2, neighborList,numNeighbors,weightMatrix,rotations,carapWeight)
        return carapWeight*cellEnergies
    @staticmethod
    def backward(ctx,grad_output):
        xyz1, xyz2, neighborList, numNeighbors, weightMatrix, rotations, carapWeight = ctx.saved_tensors
        gradxyz = grad_output.clone()
        gradxyz = gradxyz.cuda().contiguous()
        carap_cuda.backward(xyz1, xyz2, neighborList,numNeighbors,weightMatrix,rotations,gradxyz)
        return None,carapWeight*gradxyz,None,None,None,None,None,None


class CArap(torch.nn.Module):
    def forward(self, xyz1, xyz2,neighborList,numNeighbors,weightMatrix,alpha,area,carapWeight):
        return CArapFunction.apply(xyz1, xyz2,neighborList,numNeighbors,weightMatrix,alpha,area,carapWeight)
