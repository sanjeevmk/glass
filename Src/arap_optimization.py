import sys
sys.path.append("../")
from Utils import Obj
from sklearn.neighbors import NearestNeighbors
from extension.arap.cuda.arap import Arap
from extension.arap_closed.cuda.arap import ClosedArap as ArapSolveRhs
from extension.grad_arap.cuda.arap import ArapGrad
from extension.bending.cuda.arap import Bending
import numpy as np
import torch
import header
import os
import time
import trimesh
def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.mean(np.abs(array - np.expand_dims(value,0)),1).argmin()
    return idx

hi_arap = Arap()
hi_arap_solve_rhs = ArapSolveRhs()
bending = Bending()
from sklearn.neighbors import NearestNeighbors
clf = NearestNeighbors(n_neighbors=1,p=1,n_jobs=15)

'''
def norm_points(vertices,ref_verts):
    bmin = np.min(ref_verts,axis=0)
    bmax = np.max(ref_verts,axis=0)
    diagsq = np.sum(np.power(bmax-bmin,2))
    ref_diag = np.sqrt(diagsq)

    bmin = np.min(vertices,axis=0)
    bmax = np.max(vertices,axis=0)
    diagsq = np.sum(np.power(bmax-bmin,2))
    diag = np.sqrt(diagsq)

    s = np.eye(3)
    s *= (ref_diag/diag)
    vertices = np.dot(vertices,s)

    bmin = np.min(ref_verts,axis=0)
    bmax = np.max(ref_verts,axis=0)
    ref_bcenter = (bmin+bmax)/2.0

    bmin = np.min(vertices,axis=0)
    bmax = np.max(vertices,axis=0)
    bcenter = (bmin+bmax)/2.0
    #vertices = vertices + (ref_bcenter-bcenter)

    return vertices
'''
def norm_points(vertices):
    bmin = np.min(vertices,axis=0)
    bmax = np.max(vertices,axis=0)
    diagsq = np.sum(np.power(bmax-bmin,2))
    diag = np.sqrt(diagsq)

    s = np.eye(3)
    s *= (1.0/diag)
    vertices = np.dot(vertices,s)

    bmin = np.min(vertices,axis=0)
    bmax = np.max(vertices,axis=0)
    bcenter = (bmin+bmax)/2.0
    vertices -= bcenter

    return vertices

def project(sourceShape,perturbedShape):
    source_obj = Obj.Obj(sourceShape)
    source_obj.readContents()

    perturbed_obj = Obj.Obj(perturbedShape)
    perturbed_obj.readContents()
    deformed_verts,_ = perturbed_obj.readPointsAndFaces(normed=True)
    source_verts,_ = source_obj.readPointsAndFaces(normed=True)

    numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,square_weightMatrix = source_obj.getArapMetadata()

    numNeighbors = torch.unsqueeze(torch.from_numpy(numNeighbors).int().cuda(),0)
    accnumNeighbors = torch.unsqueeze(torch.from_numpy(accnumNeighbors).int().cuda(),0)
    neighborsMatrix = torch.unsqueeze(torch.from_numpy(neighborsMatrix).int().cuda(),0)
    weightMatrix = torch.unsqueeze(torch.from_numpy(weightMatrix).float().cuda(),0)
    square_weightMatrix = -1.0*torch.from_numpy(square_weightMatrix).float().cuda()
    perturbedShape = torch.unsqueeze(torch.from_numpy(deformed_verts).float().cuda(),0)
    sourceShape = torch.unsqueeze(torch.from_numpy(source_verts).float().cuda(),0)

    bestEnergy = 1e9
    projection_samples = []
    for i in range(10):
        energy,rotations = hi_arap(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,1.0)
        #updated_rotations = bending(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,rotations,1.0)
        rhs = hi_arap_solve_rhs(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,rotations,1.0)
        perturbedShape = torch.unsqueeze(torch.linalg.solve(square_weightMatrix,rhs),0)

        energy,_ = hi_arap(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,1.0)
        energy = energy.mean()

        projection_samples.append(perturbedShape.cpu().detach().numpy()[0])
        if energy.item() < bestEnergy:
            bestEnergy = energy.item()
            bestShape = perturbedShape 

        if abs(bestEnergy) < 1e-6:
            break

    print("High-res projection:",bestEnergy)
    bestObj = Obj.Obj("dummy.obj")
    bestObj.setVertices(bestShape.cpu().detach().numpy()[0])
    bestObj.setFaces(faces)
    return projection_samples,bestObj

if __name__ == '__main__':
    import sys
    source_shape = sys.argv[1]
    deformed_shape = sys.argv[2]
    out_dir = sys.argv[3]

    faces = trimesh.load(source_shape,process=False).faces

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    projection_samples,bestObj = project(source_shape,deformed_shape)

    for i in range(len(projection_samples)):
        _obj = Obj.Obj("dummy.obj")
        _obj.setVertices(projection_samples[i])
        _obj.setFaces(faces)
        _obj.saveAs(os.path.join(out_dir,str(i+1)+'.obj'))
    bestObj.saveAs(os.path.join(out_dir,str(i+1)+'.obj'))
