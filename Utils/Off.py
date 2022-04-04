import csv
import numpy as np
import sys
sys.path.append("../")
import random
random.seed(10)
class Off:
    def __init__(self,filePath):
        self.filePath = filePath
        self.numPoints = 0
        self.numFaces = 0
        self.vertices = []
        self.faces = []
        print(self.filePath)

    def readPointsAndFaces(self):
        if len(self.vertices)!=0 and len(self.faces)!=0:
            return self.vertices,self.faces
        offContents = []
        with open(self.filePath,'r') as handle:
            offContents = list(csv.reader(handle,delimiter=' '))
        self.numPoints = int(offContents[1][0])
        self.numFaces = int(offContents[1][1])
        offContents = offContents[2:]
        pointContents = offContents[:self.numPoints]
        faceContents = offContents[self.numPoints:]
        for pointC in pointContents:
            self.vertices.append([float(coord) for coord in pointC if coord])
        for faceC in faceContents:
            self.faces.append([int(vertId) for vertId in faceC[1:]])
        self.vertices = np.array(self.vertices)

        bmin = np.min(self.vertices,axis=0)
        bmax = np.max(self.vertices,axis=0)
        diagsq = np.sum(np.power(bmax-bmin,2))
        diag = np.sqrt(diagsq)
        s = np.eye(3)
        s *= (1.0/diag)
        self.vertices = np.dot(self.vertices,s)
        #random.shuffle(self.vertices)
        self.faces = np.array(self.faces)
        return self.vertices,self.faces

    def readFaces(self):
        offContents = []
        with open(self.filePath,'r') as handle:
            offContents = list(csv.reader(handle,delimiter=' '))
        self.numPoints = int(offContents[1][0])
        self.numFaces = int(offContents[1][1])
        offContents = offContents[2:]
        faceContents = offContents[self.numPoints:]
        for faceC in faceContents:
            self.faces.append([int(vertId) for vertId in faceC[1:]])
        self.faces = np.array(self.faces)
        return self.faces

    def readPoints(self):
        offContents = []
        with open(self.filePath,'r') as handle:
            offContents = list(csv.reader(handle,delimiter=' '))
        self.numPoints = int(offContents[1][0])
        offContents = offContents[2:]
        pointContents = offContents[:self.numPoints]
        for pointC in pointContents:
            self.vertices.append([float(coord) for coord in pointC])

        self.vertices = np.array(self.vertices)

        bmin = np.min(self.vertices,axis=0)
        bmax = np.max(self.vertices,axis=0)
        diagsq = np.sum(np.power(bmax-bmin,2))
        diag = np.sqrt(diagsq)
        s = np.eye(3)
        s *= (1.0/diag)
        self.vertices = np.dot(self.vertices,s)

        return self.vertices

    def setVertices(self,vertices):
        self.vertices = vertices
        self.num_points = len(vertices)

    def setFaces(self,faces):
        self.faces = faces
        self.numFaces = len(faces)

    def savePointsAs(self,path):
        with open(path,'w') as handle:
            for vert in self.vertices[:-1]:
                handle.write(str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\r\n")
            vert = self.vertices[-1]
            handle.write("3 " + str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\r\n")

    def saveAs(self,path):
        with open(path,'w') as handle:
            handle.write("OFF\r\n")
            handle.write(str(self.num_points) + " " + str(self.numFaces) + " 0\r\n")
            for vert in self.vertices:
                handle.write(str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\r\n")
            for face in self.faces[:-1]:
                handle.write("3 " + str(face[0]) + " " + str(face[1]) + " " + str(face[2]) + "\r\n")
            face = self.faces[-1]
            handle.write("3 " + str(face[0]) + " " + str(face[1]) + " " + str(face[2]) + "\r\n")

    def getArapMetadata(self):
        return self.numNeighbors,self.neighborMatrix,self.weightMatrix

    def computeArapMetadata(self):
        from Utils.misc import angle_between,cot
        self.readPointsAndFaces()
        self.neighborMatrix = np.zeros((self.numPoints,self.numPoints),dtype='int')
        self.numNeighbors = np.zeros((self.numPoints,),dtype='int')
        self.weightMatrix = np.zeros((self.numPoints,self.numPoints),dtype='float64')

        edges = set()
        edge2Faces = {}
        for faceId,face in enumerate(self.faces):
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

        for pointId,point1 in enumerate(self.vertices):
            for neighborPointer in range(self.numNeighbors[pointId]):
                neighborId = self.neighborMatrix[pointId,neighborPointer]
                if self.weightMatrix[neighborId,pointId]!=0:
                    self.weightMatrix[pointId,neighborId] = self.weightMatrix[neighborId,pointId]
                    continue
                neighborPoint = self.vertices[neighborId]
                #assert(len(edge2Faces[(pointId,neighborId)]) <= 2)

                cotanSum = 0.0

                for sharedFaceId in edge2Faces[(pointId,neighborId)]:
                    face = self.faces[sharedFaceId]
                    oppositeVertexId = [vid for vid in face if vid!=pointId and vid!=neighborId][0]
                    oppositeVertex = self.vertices[oppositeVertexId]
                    angle = angle_between(point1-oppositeVertex,neighborPoint-oppositeVertex)
                    cotanSum += cot(angle)
                cotanSum *= 0.5
                self.weightMatrix[pointId,neighborId] = cotanSum
