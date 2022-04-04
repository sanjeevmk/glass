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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.mean(np.abs(array - np.expand_dims(value,0)),1).argmin()
    return idx

arap = Arap()
bending = Bending()
arap_solve_rhs = ArapSolveRhs()

decim_template = sys.argv[1]
high_template = sys.argv[2]
high_mesh_path = sys.argv[3]
low_deform_path = sys.argv[4]

decim_obj = Obj.Obj(decim_template)
decim_obj.readContents()

high_template_obj = Obj.Obj(high_template)
high_template_obj.readContents()

high_obj = Obj.Obj(high_mesh_path)
high_obj.readContents()

low_deform_obj = Obj.Obj(low_deform_path)
low_deform_obj.readContents()

low_verts,_ = decim_obj.readPointsAndFaces(normed=True)
high_template_verts,_ = high_template_obj.readPointsAndFaces(normed=True)
high_verts,faces = high_obj.readPointsAndFaces(normed=True)
low_deform_verts,_ = low_deform_obj.readPointsAndFaces(normed=True)

#nbrs = NearestNeighbors(n_neighbors=1).fit(high_verts)
#indices = np.squeeze(nbrs.kneighbors(low_verts,return_distance=False))
indices = []
for i in range(low_verts.shape[0]):
    idx = find_nearest(high_template_verts,low_verts[i,:])
    indices.append(idx)
indices = np.array(indices)

deformed_verts = np.array(high_verts)
deformed_verts[indices,:] = low_deform_verts

bestObj = Obj.Obj("dummy.obj")
bestObj.setVertices(deformed_verts)
bestObj.setFaces(faces)
bestObj.saveAs("deform.obj")

numNeighbors,accnumNeighbors,neighborsMatrix,weightMatrix,square_weightMatrix = high_obj.getArapMetadata()
indices_list = indices.tolist()
to_keep_indices = [i for i in range(high_verts.shape[0]) if i not in indices_list]

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
#reduced_square_matrix = square_weightMatrix[to_keep_indices,:]
#reduced_square_matrix = reduced_square_matrix[:,to_keep_indices]
#expanded_square_matrix = torch.cat([square_weightMatrix,torch.zeros(square_weightMatrix.size()[1],len(indices_list)).float().cuda()],1)
#expanded_square_matrix = torch.cat([expanded_square_matrix,torch.zeros(len(indices_list),expanded_square_matrix.size()[1]).float().cuda()],0)
#expanded_square_matrix = torch.cat([square_weightMatrix,torch.zeros(len(indices_list),square_weightMatrix.size()[1]).float().cuda()],0)
#print(expanded_square_matrix.size())
for i,vert_index in enumerate(indices_list):
    #expanded_square_matrix[vert_index,num_verts+i] = 1.0
    #expanded_square_matrix[num_verts+i,vert_index] = 1.0
    new_square_matrix[vert_index,:] = torch.zeros(new_square_matrix.size()[1]).float().cuda()
    new_square_matrix[vert_index,vert_index] = 1

bestEnergy = 1e9
for i in range(10):
    energy,rotations = arap(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,1.0)
    updated_rotations = bending(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,rotations,1.0)
    rhs = arap_solve_rhs(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,updated_rotations,1.0)
    new_rhs = rhs.clone().detach().float().cuda()
    new_rhs[indices_list,:] = low_deform_verts
    #expanded_rhs = torch.cat([rhs,low_deform_verts],0)
    #reduced_rhs = rhs[to_keep_indices,:]
    perturbedShape = torch.unsqueeze(torch.linalg.solve(new_square_matrix,new_rhs),0)
    #expanded_perturbedShape = torch.unsqueeze(torch.linalg.solve(expanded_square_matrix,expanded_rhs),0)
    #expanded_perturbedShape = torch.unsqueeze(torch.linalg.solve(expanded_square_matrix,expanded_rhs),0)
    #perturbedShape = torch.unsqueeze(expanded_perturbedShape[0,:num_verts,:num_verts],0)
    #perturbedShape = torch.unsqueeze(torch.linalg.solve(square_weightMatrix,rhs),0)
    bestObj = Obj.Obj("dummy.obj")
    bestObj.setVertices(perturbedShape.cpu().detach().numpy()[0])
    bestObj.setFaces(faces)
    bestObj.saveAs("hi_"+str(i)+".obj")

    #perturbedShape[0,to_keep_indices,:] = reduced_perturbedShape
    energy,_ = arap(sourceShape,perturbedShape,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,1.0)
    energy = energy.mean()
    print(energy)

    if energy.item() < bestEnergy:
        bestEnergy = energy.item()
        bestShape = perturbedShape 

    if abs(bestEnergy) < 1e-6:
        break
'''
if bestEnergy > 1e-6:
    bestEnergy = 1e9
    bestShape = None
    perturbedShape = torch.unsqueeze(torch.from_numpy(deformed_verts).float().cuda(),0)
    proposedShape = header.Position(perturbedShape) #.view(1,-1))
    codeOptimizer = torch.optim.Adam(proposedShape.parameters(),lr=1e-3,weight_decay=0 )

    for i in range(300):
        codeOptimizer.zero_grad()

        energy = None
        #energy,_ = arap(sourceShape,proposedShape.proposed.view(1,num_verts,3),neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,torch.tensor(1.0).float().cuda())
        energy,_ = arap(sourceShape,proposedShape.proposed,neighborsMatrix,numNeighbors,accnumNeighbors,weightMatrix,torch.tensor(1.0).float().cuda())
        energy = energy.mean()
        print(energy.item())

        energy.backward()
        codeOptimizer.step()
        if energy.item() < bestEnergy:
            bestEnergy = energy.item()
            bestShape = proposedShape.proposed #.view(1,num_verts,3)

        if bestEnergy < 1e-6:
            break

    print("perturbed optimizer",i,bestEnergy,flush=True)
'''

print(bestEnergy)
bestObj = Obj.Obj("dummy.obj")
bestObj.setVertices(bestShape.cpu().detach().numpy()[0])
bestObj.setFaces(faces)
bestObj.saveAs("hi.obj")
