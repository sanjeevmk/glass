"""
As Rigid as Possible Interpolation from a pair of Mesh structures
"""

from Escher.Geometry import Mesh
from typing import List
import numpy as np
import Escher.GeometryRoutines as geom
import Escher.AlgebraRoutines as alg
import logging
from scipy.linalg import block_diag
from scipy.spatial.transform import Slerp,Rotation

def interpolate(src_mesh:Mesh, tgt_mesh:Mesh, interval:float, fragment_resolution="quadratic") -> List[Mesh]:
    """
    Interpolate between 2 meshes which have corresponding vertices and same topology.

    Arguments:
    Source Mesh, Target Mesh, Transformation method.
    interval: specifies a float between 0 and 1.
    Transformation method = one_ring,tetrahedralize.
    """

    interpolated_meshes = []

    src_face_batch = src_mesh.get_faces_as_matrices() # fx3x3
    tgt_face_batch = tgt_mesh.get_faces_as_matrices()


    per_face_slerp_instances = []
    per_face_scales = []
    identity_rotation = np.expand_dims(np.eye(3),0)
    per_face_transformations = []
    per_face_translations = []

    for _index in range(src_face_batch.shape[0]):
        src_face_matrix = src_face_batch[_index]
        tgt_face_matrix = tgt_face_batch[_index]
        src_tet_matrix = geom.tetrahedralize(src_face_matrix)
        tgt_tet_matrix = geom.tetrahedralize(tgt_face_matrix)
        mat_Q = alg.get_transformation_matrix_for((src_tet_matrix[:3,:]-src_tet_matrix[3,:]).T,
                                                    (tgt_tet_matrix[:3,:]-tgt_tet_matrix[3,:]).T)
        face_translation = np.expand_dims(tgt_tet_matrix[3,:].T,-1) - (mat_Q @ np.expand_dims(src_tet_matrix[3,:].T,-1))
        per_face_translations.append(face_translation.squeeze())

        per_face_transformations.append(mat_Q)
                                                    
        R,S = alg.get_rotation_scale_from_transformation(mat_Q)

        rotation_endpoints_matrix = np.concatenate([identity_rotation,np.expand_dims(R,0)],axis=0)

        _slerp = Slerp(times=[0,1],rotations=Rotation.from_matrix(rotation_endpoints_matrix))

        per_face_slerp_instances.append(_slerp)
        per_face_scales.append(S)

    if fragment_resolution == "average":
        vertex_id_to_face_id = src_mesh.get_vertex_id_to_face_id()
        number_of_faces_each_vertex = np.expand_dims(np.array([len(face_list) for face_list in vertex_id_to_face_id]),-1)
        for t in np.arange(0,1+interval,interval):
            new_vertices = np.zeros(src_mesh.vertices.shape)

            for _index in range(src_face_batch.shape[0]):
                interpolated_rotation_matrix_face = per_face_slerp_instances[_index]([t])[0].as_matrix()
                interpolated_scale_matrix_face = (1-t)*np.eye(3) + t*per_face_scales[_index]
                interpolated_transformation_matrix = interpolated_rotation_matrix_face @ interpolated_scale_matrix_face
                interpolated_translation = t*per_face_translations[_index].T
                src_face_matrix = src_face_batch[_index]
                new_face_matrix = (interpolated_transformation_matrix @ src_face_matrix.T).T + interpolated_translation

                face = src_mesh.faces[_index]

                for i,vertex_id in enumerate(face):
                    new_vertices[vertex_id,:] += new_face_matrix[i,:]

            new_vertices /= number_of_faces_each_vertex
            interpolated_mesh = Mesh(vertices=new_vertices,faces=src_mesh.faces)
            interpolated_meshes.append(interpolated_mesh)
    elif fragment_resolution == "quadratic":
        src_face_inverse_list = []

        #mat_H = np.zeros((src_mesh.num_vertices-1,src_mesh.num_vertices-1))
        mat_H = np.zeros((src_mesh.num_vertices,src_mesh.num_vertices))

        fixed_vertex_id = 0 # this vertex id is fixed by linear interpolation,
                            # we don't solve for it. That is why the system has a solution.

        vertex_orders = [0,1,2]
        for face_index in range(src_face_batch.shape[0]):
            src_face_matrix = src_face_batch[face_index,:,:].T
            src_face_inverse = np.linalg.inv(src_face_matrix)
            src_face_inverse_list.append(src_face_inverse)

            face = src_mesh.faces[face_index]
            for vertex_order_in_face,v_id in enumerate(face):
                #if v_id == fixed_vertex_id:
                #    continue

                other_vertex_orders = [order for order in vertex_orders if order!=vertex_order_in_face]
                row_for_vertex = src_face_inverse[vertex_order_in_face,:]
                quadratic_term = np.sum(np.square(row_for_vertex))
                mat_H[v_id,v_id] += quadratic_term
                #mat_H[v_id-1,v_id-1] += quadratic_term

                for other_vertex_order_ in other_vertex_orders:
                    other_vertex_id = face[other_vertex_order_]
                    other_vertex_row = src_face_inverse[other_vertex_order_,:]
                    #if other_vertex_id == fixed_vertex_id:
                    #    continue
                    #else:
                    mixed_term = np.dot(row_for_vertex,other_vertex_row)
                    mat_H[v_id,other_vertex_id] += mixed_term
                    #mat_H[v_id-1,other_vertex_id-1] += mixed_term

        mat_H_inverse = np.linalg.inv(mat_H)

        x_index = 0
        y_index = 1
        z_index = 2

        src_fixed_vertex = np.expand_dims(src_mesh.vertices[fixed_vertex_id],0)
        tgt_fixed_vertex = np.expand_dims(tgt_mesh.vertices[fixed_vertex_id],0)

        for t in np.arange(0,1,interval):
            #print(t,flush=True)
            mat_Gx = np.zeros((src_mesh.num_vertices,1))
            #mat_Gx = np.zeros((src_mesh.num_vertices-1,1))
            mat_Gy = np.zeros((src_mesh.num_vertices,1))
            #mat_Gy = np.zeros((src_mesh.num_vertices-1,1))
            mat_Gz = np.zeros((src_mesh.num_vertices,1))
            #mat_Gz = np.zeros((src_mesh.num_vertices-1,1))

            interpolated_fixed_vertex = ((1-t)*src_fixed_vertex + t*tgt_fixed_vertex)

            for face_index in range(src_face_batch.shape[0]):
                interpolated_rotation_matrix_face = per_face_slerp_instances[face_index]([t])[0].as_matrix()
                interpolated_scale_matrix_face = (1-t)*np.eye(3) + t*per_face_scales[face_index]
                interpolated_transformation_matrix = interpolated_rotation_matrix_face @ interpolated_scale_matrix_face
                face_inverse_matrix = src_face_inverse_list[face_index]

                face = src_mesh.faces[face_index]
                for vertex_order_in_face,v_id in enumerate(face):
                    if v_id == fixed_vertex_id:
                        continue
                    linear_term_x = np.dot(face_inverse_matrix[vertex_order_in_face,:],interpolated_transformation_matrix[x_index,:])
                    mat_Gx[v_id] += -1*linear_term_x
                    #mat_Gx[v_id-1] += -1*linear_term_x
                    linear_term_y = np.dot(face_inverse_matrix[vertex_order_in_face,:],interpolated_transformation_matrix[y_index,:])
                    mat_Gy[v_id] += -1*linear_term_y
                    #mat_Gy[v_id-1] += -1*linear_term_y
                    linear_term_z = np.dot(face_inverse_matrix[vertex_order_in_face,:],interpolated_transformation_matrix[z_index,:])
                    mat_Gz[v_id] += -1*linear_term_z
                    #mat_Gz[v_id-1] += -1*linear_term_z

                    '''
                    other_vertex_orders = [order for order in vertex_orders if order!=vertex_order_in_face]

                    row_for_vertex = face_inverse_matrix[vertex_order_in_face,:]
                    for other_vertex_order_ in other_vertex_orders:
                        other_vertex_id = face[other_vertex_order_]
                        other_vertex_row = face_inverse_matrix[other_vertex_order_,:]
                        if other_vertex_id == fixed_vertex_id:
                            fixed_term_x = 2*interpolated_fixed_vertex[0][0]*row_for_vertex[0]*other_vertex_row[0]
                            fixed_term_y = 2*interpolated_fixed_vertex[0][1]*row_for_vertex[1]*other_vertex_row[1]
                            fixed_term_z = 2*interpolated_fixed_vertex[0][2]*row_for_vertex[2]*other_vertex_row[2]

                            mat_Gx[v_id] += fixed_term_x
                            #mat_Gx[v_id-1] += fixed_term_x
                            mat_Gy[v_id] += fixed_term_y
                            #mat_Gy[v_id-1] += fixed_term_y
                            mat_Gz[v_id] += fixed_term_z
                            #mat_Gz[v_id-1] += fixed_term_z
                    '''

            mat_G = np.hstack([mat_Gx,mat_Gy,mat_Gz])

            interpolated_vertices = -1* (mat_H_inverse @ mat_G)
            interpolated_translation = (1-t)*src_mesh.vertices + t*tgt_mesh.vertices #np.expand_dims(interpolated_fixed_vertex[0] - src_mesh.vertices[fixed_vertex_id],0) #t*tgt_mesh.vertices[fixed_vertex_id] + (1-t)* src_mesh.vertices[fixed_vertex_id]
            #interpolated_translation = t*(tgt_mesh.vertices[fixed_vertex_id+1:,:] - src_mesh.vertices[fixed_vertex_id+1:,:])
            interpolated_vertices += interpolated_translation

            #interpolated_vertices = np.vstack([interpolated_fixed_vertex,other_interpolated_vertices])
            interpolated_mesh = Mesh(vertices=interpolated_vertices,faces=src_mesh.faces)
            interpolated_meshes.append(interpolated_mesh)
    else:
        logging.error("Given fragment resolution method unknown")

    return interpolated_meshes
