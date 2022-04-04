import high_res_projection
import sys
import os

decim_template = sys.argv[1]
high_template = sys.argv[2]
high_mesh_folder = sys.argv[3]
low_deform_folder = sys.argv[4]
high_deform_folder = sys.argv[5]

if not os.path.exists(high_deform_folder):
    os.makedirs(high_deform_folder)

high_mesh_prefix = [x.split('.')[0] for x in os.listdir(high_mesh_folder)]

index = 0
for f in os.listdir(low_deform_folder):
    print(index)
    index+=1
    if not f.endswith('.obj'):
        continue

    deform_path = os.path.join(low_deform_folder,f)
    high_res_path = [x for x in high_mesh_prefix if x in f][0]
    high_res_path += '.obj'
    high_res_path = os.path.join(high_mesh_folder,high_res_path)

    high_frame_obj = high_res_projection.project(decim_template,high_template,high_res_path,deform_path)

    high_frame_obj.saveAs(os.path.join(high_deform_folder,f))

