import csv
import numpy as np
import sys
import trimesh
sys.path.append("../")
import random
random.seed(10)
class Obj:
    def __init__(self,filePath):
        self.filePath = filePath
        self.numPoints = 0
        self.numFaces = 0
        self.vertices = []
        self.faces = []

    def readPointsAndFaces(self,normed=False):
        vertices = self.readPoints(normed=normed)
        faces = self.readFaces()
        return vertices,faces


    def readFaces(self):
        if len(self.faces)!=0:
            return self.faces

        for line in self.contents:
            if line.startswith("f "):
                str_vids = line.split(" ")[1:]
                triangle_vids = [int(i)-1 for i in str_vids if i]
                self.faces.append(triangle_vids)
        self.numFaces = len(self.faces)
        self.faces= np.array(self.faces)
        return self.faces

    def readContents(self):
        contents = []
        with open(self.filePath,'r') as handle:
            contents = [line.strip() for line in handle if line.strip()]
        self.contents = contents

    def samplePoints(self,npoints):
        verts,faces = self.readPointsAndFaces(normed=True)
        trimesh_mesh = trimesh.Trimesh(verts,faces,process=False)
        sampled_points,_ = trimesh.sample.sample_surface(trimesh_mesh,npoints)

        return sampled_points

    def readPoints(self,normed=False):
        if len(self.vertices)!=0:
            return self.vertices

        for line in self.contents:
            if line.startswith("v "):
                str_coords = line.split(" ")[1:]
                coord = [float(c) for c in str_coords if c]
                self.vertices.append(coord)
        self.numPoints = len(self.vertices)

        self.vertices = np.array(self.vertices)

        if normed:
            bmin = np.min(self.vertices,axis=0)
            bmax = np.max(self.vertices,axis=0)
            bcenter = (bmin+bmax)/2.0
            self.vertices -= bcenter
            bmin = np.min(self.vertices,axis=0)
            bmax = np.max(self.vertices,axis=0)
            diagsq = np.sum(np.power(bmax-bmin,2))
            diag = np.sqrt(diagsq)
            s = np.eye(3)
            s *= (1.0/diag)
            normed_vertices = np.dot(self.vertices,s)
            bmin = np.min(self.vertices,axis=0)
            bmax = np.max(self.vertices,axis=0)
            bcenter = (bmin+bmax)/2.0
            self.vertices -= bcenter

            self.vertices = normed_vertices
        return self.vertices

    def setVertices(self,vertices):
        self.vertices = vertices
        self.numPoints = len(vertices)

    def setFaces(self,faces):
        #if not faces.all():
        #    faces+=1
        self.faces = faces
        self.numFaces = len(faces)

    def savePointsAs(self,path):
        with open(path,'w') as handle:
            for vert in self.vertices[:-1]:
                handle.write(str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\r\n")
            vert = self.vertices[-1]
            handle.write(str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\r\n")

    def saveAs(self,path):
        with open(path,'w') as handle:
            for vert in self.vertices:
                handle.write("v " + str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\r\n")
            for face in self.faces[:-1]:
                handle.write("f " + str(face[0]+1) + " " + str(face[1]+1) + " " + str(face[2]+1) + "\r\n")
            face = self.faces[-1]
            handle.write("f " + str(face[0]+1) + " " + str(face[1]+1) + " " + str(face[2]+1) + "\r\n")

    def getArapMetadata(self):
        if hasattr(self,'neighborMatrix'):
            return self.numNeighbors,self.accnumNeighbors,self.neighborMatrix,self.weightMatrix,self.square_weightMatrix
        else:
            self.computeArapMetadata()
            return self.numNeighbors,self.accnumNeighbors,self.neighborMatrix,self.weightMatrix,self.square_weightMatrix

    def computeArapMetadata(self):
        #if hasattr(self,'neighborMatrix'):
        #    return self.numNeighbors,self.accnumNeighbors,self.neighborMatrix,self.weightMatrix,np.array([len(self.weightMatrix)]),self.square_weightMatrix

        from Utils.misc import angle_between,cot,tan
        vertices,faces = self.readPointsAndFaces(normed=True)
        self.numNeighbors = np.zeros((self.numPoints,),dtype='int')
        self.accnumNeighbors = np.zeros((self.numPoints,),dtype='int')
        self.square_weightMatrix = np.zeros((self.numPoints,self.numPoints),dtype='float64')

        edges = set()
        edge2Faces = {}
        neighlist = {}
        for faceId,face in enumerate(faces):
            vid0 = face[0] ; vid1 = face[1] ; vid2 = face[2]
            if (vid0,vid1) not in edges and (vid1,vid0) not in edges:
                edges.add((vid0,vid1))
                self.numNeighbors[vid0] += 1
                self.numNeighbors[vid1] += 1

                if vid0 not in neighlist:
                    neighlist[vid0] = []
                if vid1 not in neighlist:
                    neighlist[vid1] = []
                neighlist[vid0].append(vid1)
                neighlist[vid1].append(vid0)

            if (vid0,vid2) not in edges and (vid2,vid0) not in edges:
                edges.add((vid0,vid2))
                self.numNeighbors[vid0] += 1
                self.numNeighbors[vid2] += 1

                if vid0 not in neighlist:
                    neighlist[vid0] = []
                if vid2 not in neighlist:
                    neighlist[vid2] = []
                neighlist[vid0].append(vid2)
                neighlist[vid2].append(vid0)

            if (vid1,vid2) not in edges and (vid2,vid1) not in edges:
                edges.add((vid1,vid2))
                self.numNeighbors[vid1] += 1
                self.numNeighbors[vid2] += 1

                if vid1 not in neighlist:
                    neighlist[vid1] = []
                if vid2 not in neighlist:
                    neighlist[vid2] = []
                neighlist[vid1].append(vid2)
                neighlist[vid2].append(vid1)

            if (vid0,vid1) not in edge2Faces and (vid1,vid0) not in edge2Faces:
                edge2Faces[(vid0,vid1)] = [faceId]
                edge2Faces[(vid1,vid0)] = [faceId]
            else:
                edge2Faces[(vid0,vid1)].append(faceId)
                edge2Faces[(vid1,vid0)].append(faceId)

            if (vid0,vid2) not in edge2Faces and (vid2,vid0) not in edge2Faces:
                edge2Faces[(vid0,vid2)] = [faceId]
                edge2Faces[(vid2,vid0)] = [faceId]
            else:
                edge2Faces[(vid0,vid2)].append(faceId)
                edge2Faces[(vid2,vid0)].append(faceId)

            if (vid1,vid2) not in edge2Faces and (vid2,vid1) not in edge2Faces:
                edge2Faces[(vid1,vid2)] = [faceId]
                edge2Faces[(vid2,vid1)] = [faceId]
            else:
                edge2Faces[(vid1,vid2)].append(faceId)
                edge2Faces[(vid2,vid1)].append(faceId)

        for vid in range(1,len(vertices)):
            acc_num_neighbors = np.sum(self.numNeighbors[:vid])
            self.accnumNeighbors[vid] = acc_num_neighbors
        self.neighborMatrix = np.zeros((self.accnumNeighbors[-1]+self.numNeighbors[-1]),dtype='int')
        self.weightMatrix = np.zeros((self.accnumNeighbors[-1]+self.numNeighbors[-1]),dtype='float64')

        for vid in range(len(vertices)):
            self.neighborMatrix[self.accnumNeighbors[vid]:self.accnumNeighbors[vid]+self.numNeighbors[vid]] = neighlist[vid]

        for pointId,point1 in enumerate(vertices):
            point_weight_sum = 0.0
            for neighborPointer in range(self.accnumNeighbors[pointId], self.accnumNeighbors[pointId]+self.numNeighbors[pointId]):
                neighborId = self.neighborMatrix[neighborPointer]
                neighborPoint = vertices[neighborId]
                assert(len(edge2Faces[(pointId,neighborId)]) <= 2)

                cotanSum = 0.0
                '''
                for sharedFaceId in edge2Faces[(pointId,neighborId)]:
                    face = faces[sharedFaceId]
                    oppositeVertexId = [vid for vid in face if vid!=pointId and vid!=neighborId][0]
                    oppositeVertex = vertices[oppositeVertexId]
                    angle = angle_between(point1-oppositeVertex,neighborPoint-oppositeVertex)
                    cotanSum += cot(angle)
                cotanSum *= 0.5

                self.cotan_weightMatrix[pointId,neighborId] = cotanSum
                '''
                for sharedFaceId in edge2Faces[(pointId,neighborId)]:
                    face = faces[sharedFaceId]
                    oppositeVertexId = [vid for vid in face if vid!=pointId and vid!=neighborId][0]
                    oppositeVertex = vertices[oppositeVertexId]
                    angle = angle_between(point1-oppositeVertex,point1-neighborPoint)
                    cotanSum += tan(angle/2.0)
                cotanSum /= np.linalg.norm(neighborPoint-point1)
                self.weightMatrix[neighborPointer] = cotanSum
                self.square_weightMatrix[pointId,neighborId] = cotanSum
                point_weight_sum += cotanSum
            self.square_weightMatrix[pointId,pointId] = -1 * point_weight_sum

        #return self.numNeighbors,self.accnumNeighbors,self.neighborMatrix,self.weightMatrix,np.array([len(self.weightMatrix)]),self.square_weightMatrix

    def computeOrigArapMetadata(self):
        if hasattr(self,'neighborMatrix'):
            return self.numNeighbors,self.neighborMatrix,self.weightMatrix
        from Utils.misc import angle_between,cot,tan
        vertices,faces = self.readPointsAndFaces(normed=True)
        self.neighborMatrix = np.zeros((self.numPoints,self.numPoints),dtype='int')
        self.numNeighbors = np.zeros((self.numPoints,),dtype='int')
        self.weightMatrix = np.zeros((self.numPoints,self.numPoints),dtype='float64')
        self.cotan_weightMatrix = np.zeros((self.numPoints,self.numPoints),dtype='float64')

        edges = set()
        edge2Faces = {}
        for faceId,face in enumerate(faces):
            vid0 = face[0] ; vid1 = face[1] ; vid2 = face[2]
            if (vid0,vid1) not in edges and (vid1,vid0) not in edges:
                edges.add((vid0,vid1))
                self.neighborMatrix[vid0,self.numNeighbors[vid0]] = vid1
                self.neighborMatrix[vid1,self.numNeighbors[vid1]] = vid0
                self.numNeighbors[vid0] += 1
                self.numNeighbors[vid1] += 1
            if (vid0,vid2) not in edges and (vid2,vid0) not in edges:
                edges.add((vid0,vid2))
                self.neighborMatrix[vid0,self.numNeighbors[vid0]] = vid2
                self.neighborMatrix[vid2,self.numNeighbors[vid2]] = vid0
                self.numNeighbors[vid0] += 1
                self.numNeighbors[vid2] += 1
            if (vid1,vid2) not in edges and (vid2,vid1) not in edges:
                edges.add((vid1,vid2))
                self.neighborMatrix[vid1,self.numNeighbors[vid1]] = vid2
                self.neighborMatrix[vid2,self.numNeighbors[vid2]] = vid1
                self.numNeighbors[vid1] += 1
                self.numNeighbors[vid2] += 1

            if (vid0,vid1) not in edge2Faces and (vid1,vid0) not in edge2Faces:
                edge2Faces[(vid0,vid1)] = [faceId]
                edge2Faces[(vid1,vid0)] = [faceId]
            else:
                edge2Faces[(vid0,vid1)].append(faceId)
                edge2Faces[(vid1,vid0)].append(faceId)

            if (vid0,vid2) not in edge2Faces and (vid2,vid0) not in edge2Faces:
                edge2Faces[(vid0,vid2)] = [faceId]
                edge2Faces[(vid2,vid0)] = [faceId]
            else:
                edge2Faces[(vid0,vid2)].append(faceId)
                edge2Faces[(vid2,vid0)].append(faceId)

            if (vid1,vid2) not in edge2Faces and (vid2,vid1) not in edge2Faces:
                edge2Faces[(vid1,vid2)] = [faceId]
                edge2Faces[(vid2,vid1)] = [faceId]
            else:
                edge2Faces[(vid1,vid2)].append(faceId)
                edge2Faces[(vid2,vid1)].append(faceId)

        for pointId,point1 in enumerate(vertices):
            for neighborPointer in range(self.numNeighbors[pointId]):
                neighborId = self.neighborMatrix[pointId,neighborPointer]
                if self.weightMatrix[neighborId,pointId]!=0:
                    self.weightMatrix[pointId,neighborId] = self.weightMatrix[neighborId,pointId]
                    self.cotan_weightMatrix[pointId,neighborId] = self.cotan_weightMatrix[neighborId,pointId]
                    continue
                neighborPoint = vertices[neighborId]
                assert(len(edge2Faces[(pointId,neighborId)]) <= 2)

                cotanSum = 0.0
                for sharedFaceId in edge2Faces[(pointId,neighborId)]:
                    face = faces[sharedFaceId]
                    oppositeVertexId = [vid for vid in face if vid!=pointId and vid!=neighborId][0]
                    oppositeVertex = vertices[oppositeVertexId]
                    angle = angle_between(point1-oppositeVertex,neighborPoint-oppositeVertex)
                    cotanSum += cot(angle)
                cotanSum *= 0.5
                self.cotan_weightMatrix[pointId,neighborId] = cotanSum
                cotanSum = 0.0
                for sharedFaceId in edge2Faces[(pointId,neighborId)]:
                    face = faces[sharedFaceId]
                    oppositeVertexId = [vid for vid in face if vid!=pointId and vid!=neighborId][0]
                    oppositeVertex = vertices[oppositeVertexId]
                    angle = angle_between(point1-oppositeVertex,point1-neighborPoint)
                    cotanSum += tan(angle/2.0)
                cotanSum /= np.linalg.norm(neighborPoint-point1)
                self.weightMatrix[pointId,neighborId] = cotanSum
        return self.numNeighbors,self.neighborMatrix,self.weightMatrix

    def setParentPath(self,path):
        self.parentPointsPath = path

    def getParentPoints(self):
        if hasattr(self,'parentPoints'):
            return self.parentPoints
        elif hasattr(self,'parentPointsPath'):
            pts_contents = list(csv.reader(open(self.parentPointsPath,'r'),delimiter=' ',quoting=csv.QUOTE_NONNUMERIC))
            pts_contents = np.array(pts_contents)
            self.parentPoints = pts_contents
        else:
            self.parentPoints = self.vertices

        return self.parentPoints

    def getArea(self):
        if hasattr(self,'meshArea'):
            return self.meshArea

        points,faces = self.readPointsAndFaces()
        totalArea = 0.0
        for face in faces:
            vid0 = face[0] ; vid1 = face[1] ; vid2 = face[2]
            p0 = points[vid0] ; p1 = points[vid1] ; p2 = points[vid2]
            crossP = np.cross(p1-p0,p2-p0)
            faceArea = np.linalg.norm(crossP)
            totalArea += faceArea

        self.meshArea = totalArea
        return totalArea

    def getCorner(self):
        if hasattr(self,'corner'):
            return self.corner
        else:
            return -1

    def setCorner(self,corner):
        self.corner = corner

    def getLaplacian(self):
        if hasattr(self,'laplacian'):
            return self.laplacian

        points = self.readPoints()

        self.laplacian = self.cotan_weightMatrix #np.zeros((self.numPoints,self.numPoints),dtype='float64')
        #self.laplacian = np.zeros((self.numPoints,self.numPoints),dtype='float64')
        for i in range(len(points)):
            rowSum = np.sum(self.laplacian[i,:])
            self.laplacian[i,i] = -rowSum # self.numNeighbors[i]
            continue
            self.laplacian[i,i] = self.numNeighbors[i]
            for neighIndex in range(self.numNeighbors[i]):
                neighborId = self.neighborMatrix[i,neighIndex]
                self.laplacian[i,neighborId] = -1.0

        self.laplacian*=-1
        return self.laplacian

    def getEigens(self):
        if hasattr(self,'eigenvec'):
            return self.eigenCoeff,self.eigenvec

        L = self.getLaplacian()
        w,v = np.linalg.eig(L)
        wSorted = np.sort(w)
        v = v[:,w.argsort()]

        self.eigenvals = wSorted
        self.eigenvec = v

        self.eigenCoeff = np.dot(self.eigenvec.T,self.readPoints(normed=True))

        return self.eigenCoeff,self.eigenvec

    def orderByX(self):
        srcPoints = np.array(self.vertices)
        indices = np.argsort(srcPoints,0)[:,0]
        newPoints = srcPoints[indices,:]
        mapping = {index:i for i,index in enumerate(indices)}

        newFaces = []
        for f in self.faces:
            face = []
            for vid in f:
                newvid = mapping[vid]
                face.append(newvid)
            newFaces.append(face)

        newFaces = np.array(newFaces)
        self.setVertices(newPoints)
        self.setFaces(newFaces)
