import sys
import numpy as np
import os

import trimesh
from trimesh.registration import icp


dataset_folder = sys.argv[1]
template_path = sys.argv[2]
output_folder = sys.argv[3]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

templateMesh = trimesh.load(template_path,process=False)

index=0
for objF in os.listdir(dataset_folder):
    print(index)
    index+=1
    if not objF.endswith(".obj"):
        continue

    sourceMesh = trimesh.load(os.path.join(dataset_folder,objF),process=False)
    #icp_output = icp(sourceMesh.vertices,templateMesh.vertices)
    icp_output = icp(sourceMesh.vertices,templateMesh)
    aligned_vertices = icp_output[1]
    outputMesh = trimesh.Trimesh(vertices=aligned_vertices,faces=templateMesh.faces)
    outputMesh.export(os.path.join(output_folder,objF))
