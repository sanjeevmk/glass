import math
import numpy as np
def angle_between(vector_a, vector_b):
    #print(np.linalg.norm(vector_a),np.linalg.norm(vector_b))
    costheta = vector_a.dot(vector_b) / (np.linalg.norm(vector_a)*np.linalg.norm(vector_b))
    if math.isnan(costheta):
        print(costheta)
    return math.acos(costheta)

def cot(theta):
    return math.cos(theta) / math.sin(theta)

def tan(theta):
    return math.sin(theta) / math.cos(theta)

def rotateX(verts,angle):
    angle = math.radians(angle)
    matrix = np.eye(3)
    matrix[1,1] = math.cos(angle) ; matrix[1,2] = math.sin(angle)
    matrix[2,1] = -math.sin(angle) ; matrix[2,2] = math.cos(angle)

    verts = verts.dot(matrix)

    return verts