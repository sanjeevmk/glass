"""
This module holds some routines for common algebraic tasks
"""

import numpy as np
import logging
from typing import Tuple,List
from scipy.spatial.transform import Rotation

def is_rotation_matrix(R):
    # square matrix test
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False

    should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], np.float))
    should_be_one = np.allclose(np.linalg.det(R), 1)
    return should_be_identity and should_be_one

def get_translation_for(points_1: np.array, points_2: np.array) -> np.array:
    """
    This function computes the translation between 2 given sets of points, by
    computing the difference in their mean.

    Arguments:
    points_1, points_2 as 3xN or Nx3 np.arrays

    Returns:
    Q such that Q + points_1 = points_2, of shape 1x3 or 3x1 depending on input shapes
    """

    assert (points_1.shape==points_2.shape),logging.error("Input points should have same shape")

    if points_1.shape[0]==3: # its a 3xN matrix
        Q = np.expand_dims(np.mean(points_2,1) - np.mean(points_1,1),-1)
    if points_1.shape[1]==3: # its a Nx3 matrix
        Q = np.expand_dims(np.mean(points_2,0) - np.mean(points_1,0),0)

    return Q

def get_transformation_matrix_for(points_1: np.array, points_2: np.array) -> np.array:
    """
    This function computes the transformation matrix that maps points_1 to points_2.
    Both point sets should be given as 3xN arrays

    Arguments:
    points_1, points_2 of the same shape of 3xN

    Returns:
    Matrix Q, such that Q * points_1 = points_2 
    """

    assert points_1.shape == points_2.shape, logging.error("The two matrices have different shape")

    if points_1.shape[0] == points_1.shape[1]:
        # Inverse for square matrix
        m1_inverse = np.linalg.inv(points_1)
    else:
        # Moore-Penrose pseudo-inverse for rectangular matrix
        m1_inverse = np.linalg.pinv(points_1)

    matrix_Q = np.dot(points_2,m1_inverse)

    return matrix_Q

def get_rotation_scale_from_transformation(matrix: np.array) -> Tuple[np.array,np.array] :
    """
    This function breaks the given transformation matrix into a Rotation matrix and a Scale matrix
    as described in "As-Rigid-As-Possible Shape Interpolation" by Alexa et al

    Arguments:
    matrix : Any transformation matrix

    Returns:
    R_gamma: Rotation matrix 3x3
    S: Scale matrix 3x3
    """

    R_alpha,D,R_beta = np.linalg.svd(matrix,full_matrices=True)

    D = np.eye(3)*D

    R_gamma = R_alpha @ R_beta
    if np.linalg.det(R_gamma) < 0:
        R_gamma[0,:] *= -1

    S = R_beta.T @ D @ R_beta

    assert is_rotation_matrix(R_gamma), logging.error("Computed matrix is not a rotation")
    return (R_gamma,S)

def rotation_matrix_to_quaternion(matrix: np.array) -> np.array:
    """
    This function takes a 3x3 rotation matrix and transforms it to quaternion representation
    in the form [x,y,z,w] as a numpy array of 4 scalar values.

    Arguments:
    matrix as 3x3 np array

    Returns:
    Quaternion as 1x4 np array
    """

    assert is_rotation_matrix(matrix), logging.error("Input matrix is not a rotation")

    _rot = Rotation.from_matrix(matrix)
    quat = _rot.as_quat()

    return quat

def quaternion_to_rotation_matrix(quaternion: np.array) -> np.array:
    """
    This function takes a quaternion of the form [x,y,z,w] as a numpy array of 4 scalars
    and returns its equivalent 3x3 rotation matrix.

    Arguments:
    quaternion as 1x4 np array 

    Returns:
    Rotation as a 3x3 matrix 
    """

    _rot = Rotation.from_quat(quaternion)
    R = _rot.as_matrix()

    return R

def batch_quaternions_to_rotation_matrices(batch_quaternions: np.array) -> List[np.array]:
    """
    This function takes a batch of quaternions, usually over all vertices or triangles
    and returns a list of their corresponding rotation matrices.

    Arguments:
    batch_quaternions: Nx4 shape

    Returns:
    batch_rotation_matrices : List of 3x3 matrices
    """

    batch_rotation_matrices = []
    for i in range(batch_quaternions.shape[0]):
        R = quaternion_to_rotation_matrix(batch_quaternions[i,:])
        assert is_rotation_matrix(R), logging.error("Computed matrix is not a rotation")
        batch_rotation_matrices.append(R)

    return batch_rotation_matrices