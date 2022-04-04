from Escher import arap_interpolate 
from Escher.Geometry import Mesh
from time import time

import os
import argparse
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument("src_dir",help="Directory containing all meshes",type=str)
parser.add_argument("out_dir",help="Directory containing output interpolations",type=str)
parser.add_argument("interval",help="Interpolation interval between 0 and 1",type=float)
#parser.add_argument("num_shapes",help="Interpolation interval between 0 and 1",type=float)

args = parser.parse_args()
src_directory = args.src_dir
#num_shapes = args.num_shapes

out_directory = args.out_dir
if not os.path.exists(out_directory):
    os.makedirs(out_directory)

mesh_list = os.listdir(src_directory)
num_pairs = len(mesh_list)*(len(mesh_list)-1)
#per_pair_shapes = num_shapes/num_pairs
#interval = float(1.0/per_pair_shapes)
interval = args.interval

pairs = []
for i,src_mesh_name in enumerate(mesh_list):
    for j,tgt_mesh_name in enumerate(mesh_list):
        if i==j:
            continue
        if (j,i) in pairs:
            continue
        pairs.append((i,j))

triplets = []
for pair in pairs:
    for i,src_mesh_name in enumerate(mesh_list):
        if i not in pair:
            if set(pair+(i,)) not in triplets:
                triplets.append(set(pair+(i,)))
triplets = [list(x) for x in triplets]

for tripindex,triplet in enumerate(triplets):
    print(triplet,flush=True)
    start = time()
    src_mesh_name = mesh_list[triplet[0]]
    tgt_mesh_name = mesh_list[triplet[1]]

    src_path = os.path.join(src_directory,src_mesh_name)
    tgt_path = os.path.join(src_directory,tgt_mesh_name)

    src_head = src_mesh_name.split(".")[0]
    tgt_head = tgt_mesh_name.split(".")[0]

    mesh1 = Mesh(src_path)
    mesh2 = Mesh(tgt_path)
    mesh1.load() ; mesh2.load()

    interpolated_meshes_p1 = arap_interpolate(mesh1,mesh2,interval,'quadratic')

    for i,mesh in enumerate(interpolated_meshes_p1):
        mesh.export(os.path.join(args.out_dir,src_head+'_'+tgt_head+'_'+str(i)+'.obj'))

    end = time()
    print("Finished p1",flush=True)
    print((end-start))

    start = time()
    src_mesh_name = mesh_list[triplet[0]]
    tgt_mesh_name = mesh_list[triplet[2]]

    src_path = os.path.join(src_directory,src_mesh_name)
    tgt_path = os.path.join(src_directory,tgt_mesh_name)

    src_head = src_mesh_name.split(".")[0]
    tgt_head = tgt_mesh_name.split(".")[0]

    mesh1 = Mesh(src_path)
    mesh2 = Mesh(tgt_path)
    mesh1.load() ; mesh2.load()

    interpolated_meshes_p2 = arap_interpolate(mesh1,mesh2,interval,'quadratic')

    for i,mesh in enumerate(interpolated_meshes_p2):
        mesh.export(os.path.join(args.out_dir,src_head+'_'+tgt_head+'_'+str(i)+'.obj'))

    end = time()

    print("Finished p2",flush=True)
    print((end-start))

    start = time()
    src_mesh_name = mesh_list[triplet[1]]
    tgt_mesh_name = mesh_list[triplet[2]]

    src_path = os.path.join(src_directory,src_mesh_name)
    tgt_path = os.path.join(src_directory,tgt_mesh_name)

    src_head = src_mesh_name.split(".")[0]
    tgt_head = tgt_mesh_name.split(".")[0]

    mesh1 = Mesh(src_path)
    mesh2 = Mesh(tgt_path)
    mesh1.load() ; mesh2.load()

    interpolated_meshes_p3 = arap_interpolate(mesh1,mesh2,interval,'quadratic')

    for i,mesh in enumerate(interpolated_meshes_p3):
        mesh.export(os.path.join(args.out_dir,src_head+'_'+tgt_head+'_'+str(i)+'.obj'))

    end = time()

    print("Finished p3",flush=True)
    print((end-start))

    start = time()

    for j,pair in enumerate(zip(interpolated_meshes_p1,interpolated_meshes_p2)):
        m1,m2 = pair
        interpolated_meshes_p1p2 = arap_interpolate(m1,m2,interval,'quadratic')

        for i,mesh in enumerate(interpolated_meshes_p1p2):
            mesh.export(os.path.join(args.out_dir,'triplet'+str(tripindex) + '_p1p2_' + str(j) + '_'+str(i)+'.obj'))

    end = time()

    print("Finished p1p2",flush=True)
    print(end-start)

    start = time()

    for j,pair in enumerate(zip(interpolated_meshes_p1,interpolated_meshes_p3)):
        m1,m2 = pair
        interpolated_meshes_p1p3 = arap_interpolate(m1,m2,interval,'quadratic')

        for i,mesh in enumerate(interpolated_meshes_p1p3):
            mesh.export(os.path.join(args.out_dir,'triplet'+str(tripindex) + '_p1p3_' + str(j) + '_'+str(i)+'.obj'))

    end = time()

    print("Finished p1p3",flush=True)
    print(end-start)

    start = time()

    for j,pair in enumerate(zip(interpolated_meshes_p2,interpolated_meshes_p3)):
        m1,m2 = pair
        interpolated_meshes_p2p3 = arap_interpolate(m1,m2,interval,'quadratic')

        for i,mesh in enumerate(interpolated_meshes_p2p3):
            mesh.export(os.path.join(args.out_dir,'triplet'+str(tripindex) + '_p2p3_' + str(j) + '_'+str(i)+'.obj'))


    end = time()

    print("Finished p2p3",flush=True)
    print(end-start)


