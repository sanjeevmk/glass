import sys
from . import Obj
import numpy as np

def rectRemesh(points,faces,corner_vid=-1):
    faces = np.array(faces)
    vertNeighbors = {}
    vertToFaces = {}
    faceEdges = {}
    for fid,face in enumerate(faces):
        v1 = face[0] ; v2 = face[1] ; v3 = face[2]

        if v1 not in vertNeighbors:
            vertNeighbors[v1] = []
        if v2 not in vertNeighbors:
            vertNeighbors[v2] = []
        if v3 not in vertNeighbors:
            vertNeighbors[v3] = []

        if v2 not in vertNeighbors[v1]:
            vertNeighbors[v1].append(v2)
        if v3 not in vertNeighbors[v1]:
            vertNeighbors[v1].append(v3)

        if v1 not in vertNeighbors[v2]:
            vertNeighbors[v2].append(v1)
        if v3 not in vertNeighbors[v2]:
            vertNeighbors[v2].append(v3)

        if v1 not in vertNeighbors[v3]:
            vertNeighbors[v3].append(v1)
        if v2 not in vertNeighbors[v3]:
            vertNeighbors[v3].append(v2)

        if v1 not in vertToFaces:
            vertToFaces[v1] = []
        vertToFaces[v1].append((fid,face))

        if v2 not in vertToFaces:
            vertToFaces[v2] = []
        vertToFaces[v2].append((fid,face))

        if v3 not in vertToFaces:
            vertToFaces[v3] = []
        vertToFaces[v3].append((fid,face))

        faceEdges[fid] = [(v1,v2),(v1,v3),(v2,v3)]

    adjacentFaces = {}
    for fid1,face1 in enumerate(faces):
        edges1 = faceEdges[fid1]
        adjacentFaces[fid1] = []
        for fid2,face2 in enumerate(faces):
            if fid1==fid2:
                continue
            edges2 = faceEdges[fid2]

            for e1 in edges1:
                rev_e1 = (e1[1],e1[0])
                for e2 in edges2:
                    if e1==e2 or rev_e1==e2:
                        adjacentFaces[fid1].append(fid2)


    if corner_vid is -1:
        corners = []
        for k in vertNeighbors:
            if len(vertNeighbors[k]) == 2:
                corners.append((k,points[k]))
        smallestX = 1e9
        for corner in corners:
            if corner[1][0] < smallestX:
                smallestX = corner[1][0]
                corner_vid = corner[0]

    reorderedPoints = []
    current_face = vertToFaces[corner_vid][0][0]
    reorderedPoints.append(points[corner_vid])
    idmap = {}
    idmap[corner_vid] = 0
    doneFaces = []
    doneVerts = [corner_vid]

    while len(reorderedPoints)!=len(points):
        for vid in faces[current_face]:
            if vid in doneVerts:
                continue
            idmap[vid] = len(reorderedPoints)
            reorderedPoints.append(points[vid])
            doneVerts.append(vid)
        doneFaces.append(current_face)

        for f in adjacentFaces[current_face]:
            if f not in doneFaces:
                current_face = f
                break

    newFaces = []
    for f in faces:
        newf = []
        for vid in f:
            newf.append(idmap[vid])
        newFaces.append(newf)

    return np.array(reorderedPoints),np.array(newFaces),corner_vid

if __name__ == "__main__":
    objf = sys.argv[1]
    obj = Obj.Obj(objf)
    obj.readContents()
    points,faces = obj.readPointsAndFaces()
    newPoints,newFaces,corner_vid = rectRemesh(points,faces)
    obj.setVertices(newPoints)
    obj.setFaces(newFaces)
    obj.saveAs("test.obj")
             
