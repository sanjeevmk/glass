from Escher.Geometry import Mesh
from Escher import ArapInterpolate 
from typing import List
import argparse
import os

def deformation_interpolation(src_mesh_path:str, tgt_mesh_path:str, interval:float=0.1,fragment_resolution:str='average') -> List[Mesh]:
    """
    This function is a wrapper to deformation interpolation between two given meshes.

    Arguments:
    src_mesh_path, tgt_mesh_path : Paths on disk to two meshes.
    interval : A float value between 0 and 1, specifying the spacing between interpolated meshes.
    fragment_resolution: Either of "average" or "quadratic". This determines how the method resolves the vertex positions
    when each triangle of source is independently interpolated.
    """
    src_mesh = Mesh(src_mesh_path)
    tgt_mesh = Mesh(tgt_mesh_path)

    src_mesh.load()
    tgt_mesh.load()

    _interpolated_meshes = ArapInterpolate.interpolate(src_mesh,tgt_mesh,interval,fragment_resolution=fragment_resolution)
    return _interpolated_meshes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_mesh_path",help="Source mesh path to start interpolation",type=str)
    parser.add_argument("tgt_mesh_path",help="Target mesh path to end interpolation",type=str)
    parser.add_argument("destination_folder",help="Directory to store the interpolated meshes",type=str)
    parser.add_argument("--interval",help="Float value between 0 and 1, specifying interpolation spacing",type=float,default=0.1)
    parser.add_argument("--fragment_resolution",help="If each triangle is independently interpolated, how should the vertex \
                                                    positions be resolved",type=str,default='average',choices=['average','quadratic'])

    args = parser.parse_args()
    interpolated_meshes = interpolate_from_paths(args.src_mesh_path,args.tgt_mesh_path,args.interval,args.fragment_resolution)

    if not os.path.exists(args.destination_folder):
        os.makedirs(args.destination_folder)

    for i,m in enumerate(interpolated_meshes):
        m.export(os.path.join(args.destination_folder,str(i)+'.obj'))
