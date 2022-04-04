""" This module defines 3D geometry structures """

import trimesh
import logging
import sys
import numpy as np
from typing import List

class Mesh(trimesh.Trimesh):
    """
    Mesh structure, that extends trimesh.Trimesh to add some custom functions

    Attributes:
        mesh_path: The full path to where the mesh is (or to be) stored
    """

    def __init__(self,mesh_path="",vertices=[],faces=[],face_normals=[],vertex_normals=[],**kwargs):
        super(Mesh,self).__init__(vertices=vertices,faces=faces,**kwargs)
        self.mesh_path = mesh_path

        if len(vertices)>0 and len(faces)>0:
            self.num_vertices = len(vertices)
            self.num_faces = len(faces)
            super(Mesh,self).__init__(vertices=vertices,faces=faces,process=False,**kwargs)


    def load(self,**kwargs) -> trimesh.Trimesh:
        """
        Load trimesh object from the path attribute mesh_path
        """

        if not self.mesh_path:
            logging.error("Set the mesh_path attribute first")
            sys.exit()

        __mesh = trimesh.load_mesh(self.mesh_path,process=False)
        self.num_vertices = len(__mesh.vertices)
        self.num_faces = len(__mesh.faces)
        super(Mesh,self).__init__(vertices=__mesh.vertices,faces=__mesh.faces,process=False,**kwargs)

    def get_faces_as_matrices(self) -> np.array:
        """
        A utility to return triangular faces as a numpy array of 3x3 matrices.
        In each such 3x3 matrix representing a triangle, each row corresponds to 1 vertex of 
        the triangle.

        Returns: np.array of shape Fx3x3 where F is number of faces, and each row of
        the returned matrices correspond to a vertex
        """

        np_triangles = [] 
        for face in self.faces:
            _triangle = [] # list of 3 vertices
            for vertex_id in face:
                vertex = self.vertices[vertex_id]
                _triangle.append(vertex)
            np_triangle = np.stack(_triangle,axis=0)
            np_triangles.append(np_triangle)

        return np.array(np_triangles)

    def get_vertices_as_one_ring_matrices(self) -> List[np.array]:
        """
        A utility to return each vertex as a matrix of its 1-ring neighbors.
        The 1-ring neihborhood matrix is of mx3, where m is the number of neighbors of a vertex v_i.
        Each row corresponds to 1 vertex.

        Returns: A list of mx3 np arrays.
        """

        one_ring_list = []

        for neighbors in self.vertex_neighbors:
            one_ring_matrix = []
            for neighbor_id in neighbors:
                vertex = self.vertices[neighbor_id]
                one_ring_matrix.append(vertex)
            one_ring_list.append(np.array(one_ring_matrix))

        return one_ring_list
    
    def get_vertex_id_to_face_id(self) -> List[List]:
        """
        A utility to map vertices to the faces they belong to.

        Returns: A list of lists, where each outer index corresponds to a vertex 
        in self.vertices in the same order, and each inner list is the list of face ids
        that the vertex belongs to.
        """

        _vertex_to_faces = [[] for i in range(self.num_vertices)]
        for face_id,face in enumerate(self.faces):
            for vertex_id in face:
                _vertex_to_faces[vertex_id].append(face_id)

        return _vertex_to_faces
        
if __name__ == "__main__":
    pass
