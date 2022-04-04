from Escher.tools import deformation_interpolation
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
        print("Progress: Src Mesh="+str(i)+" Target Mesh="+str(j)+" Number of Meshes="+str(len(mesh_list)),flush=True)

        src_path = os.path.join(src_directory,src_mesh_name)
        tgt_path = os.path.join(src_directory,tgt_mesh_name)

        src_head = src_mesh_name.split(".")[0]
        tgt_head = tgt_mesh_name.split(".")[0]

        interpolated_meshes = deformation_interpolation(src_path,tgt_path,interval,'quadratic')

        for index,mesh in enumerate(interpolated_meshes):
            mesh.export(os.path.join(out_directory,src_head+"_"+tgt_head+"_"+str(index)+".obj"))

        pairs.append((i,j))
