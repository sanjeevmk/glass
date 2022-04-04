"""
This module holds utility functions for some geometric operations
"""

import numpy as np

def tetrahedralize(triangle_matrix: np.array) -> np.array:
    """
    Given a triangle in the form of a 3x3 matrix where each row represents 1 vertex of the triangle,
    return a virtual 4th vertex that would tetrahedralize this triangle.
    The 4th vertex is computed as in "Deformation Transfer for Triangle Meshes" by
    Popovic, Sumner et al.

    Arguments:
    triangle_matrix: 3x3 numpy array

    Returns:
    Tetrahedralized face as a 4x3 matrix
    """

    v_1 = triangle_matrix[0,:]
    v_2 = triangle_matrix[1,:]
    v_3 = triangle_matrix[2,:]

    v4_vector = np.cross(v_2-v_1,v_3-v_1)
    scale_factor = 1.0/np.sqrt(np.linalg.norm(v4_vector))
    v4_vector *= scale_factor

    v_4 = v_1 + v4_vector

    return np.vstack([triangle_matrix,np.expand_dims(v_4,0)])