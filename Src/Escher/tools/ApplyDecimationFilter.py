"""
Given a High-res mesh and its corresponding low-res version, this module applies
the high-to-low mapping between them, to a group of other input meshes.

This mapping works only if the group of input meshes share the same topology as the pair
of low and high-res meshes.
"""

from Escher.Geometry import Mesh
from sklearn.neighbors import NearestNeighbors
from typing import List
import numpy as np

def apply_decimation_filter(high_res_mesh: Mesh, low_res_mesh: Mesh,
                            target_high_res_meshes: List[Mesh]) -> List[Mesh]:
    """
    Given a high-res Mesh structure and its corrsponding low-res Mesh structure,
    apply the high-to-low mapping that created this pair, to given list of Mesh items.
    The given list of target meshes should posses the same topology as the given pair of
    low and high-res meshes.

    Arguments:
    high and low res meshes as Mesh instances.
    list of target high-res meshesh

    Returns:
    List of low-res meshes corrsponding to the given list of high-res meshes.
    """

    high_res_verts = high_res_mesh.vertices
    low_res_verts = low_res_mesh.vertices
    low_res_faces = low_res_mesh.faces # This is the target topology

    clf = NearestNeighbors(n_neighbors=1)
    clf.fit(high_res_verts)

    high2low_index_mapping = np.squeeze(clf.kneighbors(low_res_verts,return_distance=False))

    target_low_res_meshes = []
    for _mesh in target_high_res_meshes:
        _high_res_verts = _mesh.vertices
        _low_res_verts = _high_res_verts[high2low_index_mapping]

        output_mesh = Mesh(vertices=_low_res_verts,faces=low_res_faces)
        target_low_res_meshes.append(output_mesh)

    return target_low_res_meshes

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("high_res_mesh_path",help="Path to High resolution mesh",type=str)
    parser.add_argument("low_res_mesh_path",help="Path to Corresponding low resolution mesh",type=str)
    parser.add_argument("target_meshes_directory",help="Path to folder containing high-res meshes",type=str)
    parser.add_argument("output_directory",help="Path to folder to save output low-res meshes",type=str)

    args = parser.parse_args()

    high_res_mesh = Mesh(mesh_path=args.high_res_mesh_path)
    high_res_mesh.load()

    low_res_mesh = Mesh(mesh_path=args.low_res_mesh_path)
    low_res_mesh.load()

    _target_names = os.listdir(args.target_meshes_directory)
    _high_res_meshes_paths = [os.path.join(args.target_meshes_directory,name) for name in _target_names]
    high_res_meshes = [Mesh(mesh_path=mesh_path) for mesh_path in _high_res_meshes_paths]

    for _mesh in high_res_meshes:
        _mesh.load()

    low_res_meshes = apply_decimation_filter(high_res_mesh=high_res_mesh,low_res_mesh=low_res_mesh,
                                                target_high_res_meshes=high_res_meshes)

    if low_res_meshes:
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)

    for i,_low_mesh in enumerate(low_res_meshes):
        output_path = os.path.join(args.output_directory,_target_names[i])
        _low_mesh.export(output_path)
