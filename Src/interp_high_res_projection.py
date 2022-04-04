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

def project(decim_template,high_template,high_mesh_path,low_deform):
    decim_obj = Obj.Obj(decim_template)
    decim_obj.readContents()

    high_template_obj = Obj.Obj(high_template)
    high_template_obj.readContents()

    high_obj = Obj.Obj(high_mesh_path)
    high_obj.readContents()


    low_deform_verts = None
    if isinstance(low_deform,str):
        low_deform_obj = Obj.Obj(low_deform)
        low_deform_obj.readContents()
        low_deform_verts,_ = low_deform_obj.readPointsAndFaces(normed=True)
    else:
        low_deform_verts = low_deform

    low_verts,_ = decim_obj.readPointsAndFaces(normed=True)
    high_template_verts,_ = high_template_obj.readPointsAndFaces(normed=True)
    high_verts,faces = high_obj.readPointsAndFaces(normed=True)

    indices = []
    for i in range(low_verts.shape[0]):
        idx = find_nearest(high_template_verts,low_verts[i,:])
        indices.append(idx)
    indices = np.array(indices)
    #clf.fit(high_template_verts)
    #indices = clf.kneighbors(low_verts,return_distance=False).squeeze()
    deformed_verts = np.array(high_verts)
    deformed_verts[indices,:] = low_deform_verts

    numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,square_weightMatrix = high_obj.getArapMetadata()
    indices_list = indices.tolist()

    num_verts = high_verts.shape[0]

    numNeighbors = torch.unsqueeze(torch.from_numpy(numNeighbors).int().cuda(),0)
    accnumNeighbors = torch.unsqueeze(torch.from_numpy(accnumNeighbors).int().cuda(),0)
    neighborsMatrix = torch.unsqueeze(torch.from_numpy(neighborsMatrix).int().cuda(),0)
    weightMatrix = torch.unsqueeze(torch.from_numpy(weightMatrix).float().cuda(),0)
    square_weightMatrix = -1.0*torch.from_numpy(square_weightMatrix).float().cuda()
    perturbedShape = torch.unsqueeze(torch.from_numpy(deformed_verts).float().cuda(),0)
    sourceShape = torch.unsqueeze(torch.from_numpy(high_verts).float().cuda(),0)
    low_deform_verts =  torch.from_numpy(low_deform_verts).float().cuda()

    new_square_matrix = square_weightMatrix.clone().detach().float().cuda()

    for i,vert_index in enumerate(indices_list):
        new_square_matrix[vert_index,:] = torch.zeros(new_square_matrix.size()[1]).float().cuda()
        new_square_matrix[vert_index,vert_index] = 1

    bestEnergy = 1e9
    for i in range(10):
        energy,rotations = hi_arap(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,1.0)
        #updated_rotations = bending(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,rotations,1.0)
        rhs = hi_arap_solve_rhs(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,rotations,1.0)
        new_rhs = rhs.clone().detach().float().cuda()
        new_rhs[indices_list,:] = low_deform_verts
        perturbedShape = torch.unsqueeze(torch.linalg.solve(new_square_matrix,new_rhs),0)

        energy,_ = hi_arap(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,1.0)
        energy = energy.mean()

        if energy.item() < bestEnergy:
            bestEnergy = energy.item()
            bestShape = perturbedShape 

        if abs(bestEnergy) < 1e-6:
            break

    print("High-res projection:",bestEnergy,flush=True)
    bestObj = Obj.Obj("dummy.obj")
    bestObj.setVertices(bestShape.cpu().detach().numpy()[0])
    bestObj.setFaces(faces)
    return bestObj,bestEnergy

if __name__ == '__main__':
    import sys
    high_template = sys.argv[1]
    low_template = sys.argv[2]
    high_mesh_folder = sys.argv[3]
    low_deform_folder = sys.argv[4]
    out_deform_folder = sys.argv[5]

    if not os.path.exists(out_deform_folder):
        os.makedirs(out_deform_folder)

    for f in os.listdir(low_deform_folder):
        if not f.endswith('.obj') or not f.startswith('r5'):
            continue
        start = time.time()
        high_path = ""
        low_mesh = trimesh.load(os.path.join(low_deform_folder,f),process=False)
        low_verts = norm_points(low_mesh.vertices)

        min_dist = 1e9
        min_name = ""
        present_names = []
        present_index = []
        for name in os.listdir(high_mesh_folder):
            if name.split('.obj')[0] in f:
                high_path = os.path.join(high_mesh_folder,name)
                present_names.append(name)
                ix = f.find(name.split('.obj')[0])
                present_index.append(ix)
            '''
            hi_mesh = trimesh.load(os.path.join(high_mesh_folder,name),process=False)
            hi_verts = norm_points(hi_mesh.vertices)
            clf.fit(hi_verts)
            dist,indices = clf.kneighbors(low_verts,return_distance=True)
            if np.mean(dist) < min_dist:
                min_dist = np.mean(dist)
                min_name = name
            '''
        selected_name = ""
        if int(f.split('.obj')[0].split('_')[-1])<=15:
            min_index = min(present_index)
            selected_name = present_names[present_index.index(min_index)]
        else:
            max_index = max(present_index)
            selected_name = present_names[present_index.index(max_index)]

        high_path = os.path.join(high_mesh_folder,selected_name)
        print(f,high_path,flush=True)
        high_deform,energy = project(low_template,high_template,high_path,os.path.join(low_deform_folder,f))
        high_deform.saveAs(os.path.join(out_deform_folder,f))
        end = time.time()
        print(end-start,flush=True)
