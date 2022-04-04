import trimesh 
import numpy as np
import sys
sys.path.append("../")
import os
from Utils import Obj
from sklearn.neighbors import NearestNeighbors
import math
clf = NearestNeighbors(n_neighbors=1)
srcdir = sys.argv[1]
destdir = sys.argv[2]
srcMesh = sys.argv[3]
decimMesh = sys.argv[4]

def find_nearest(array, value):
    array = np.asarray(array)
    #idx = (np.abs(array - np.expand_dims(value,0))).argmin()
    idx = np.mean(np.abs(array - np.expand_dims(value,0)),1).argmin()
    #idx = (np.abs(array - np.expand_dims(value,0))).argmin()
    return idx

if not os.path.exists(destdir):
    os.makedirs(destdir)

files = os.listdir(srcdir)
files = list(filter(lambda x:x.endswith(".obj"),files))


srcm = trimesh.load(srcMesh,process=False)
srcm_decim = trimesh.load(decimMesh,process=False)

src_verts = srcm.vertices
decim_verts = srcm_decim.vertices
faces = srcm_decim.faces
#faces = srcm.faces

preserve_indices = []
for i in range(decim_verts.shape[0]):
    idx = find_nearest(src_verts,decim_verts[i,:])
    preserve_indices.append(idx)
#print(len(indices))
#print(indices[0].shape)
#print(indices[1].shape)
#exit()
#print(src_verts.shape)
#print(decim_verts.shape)

#preserve_indices = np.squeeze(clf.kneighbors(decim_verts,return_distance=False))

print(np.min(preserve_indices),np.max(preserve_indices))
for fil in files:
    prefix = fil.split(".")[0]
    mesh = trimesh.load(os.path.join(srcdir,fil),process=False)
    verts = mesh.vertices
    print(fil,verts.shape)
    final_verts = verts[preserve_indices,:]
    #final_verts = np.array(verts)
    objInstance = Obj.Obj("dummy.obj")
    '''
    bbcenter = np.mean(final_verts,0)
    final_verts -= bbcenter
    final_verts[:,[0,1,2]] = final_verts[:,[0,2,1]]
    angle = math.radians(-90.0)
    matrix = np.eye(3)
    matrix[0,0] = math.cos(angle) ; matrix[0,2] = math.sin(angle)
    matrix[2,0] = -math.sin(angle) ; matrix[2,2] = math.cos(angle)
    final_verts = final_verts.dot(matrix)
    '''

    objInstance.setFaces(np.array(faces))
    objInstance.setVertices(final_verts)
    objInstance.saveAs(os.path.join(destdir,prefix+'.obj'))
