import csv
import numpy as np
import sys
sys.path.append("../")
class Pts:
    def __init__(self,filePath):
        self.filePath = filePath
        self.numPoints = 0
        self.vertices = []

    def readPoints(self):
        ptsContents = []
        with open(self.filePath,'r') as handle:
            ptsContents = list(csv.reader(handle,delimiter=' ',quoting=csv.QUOTE_NONNUMERIC))
        self.numPoints = len(ptsContents)

        self.vertices = np.array(ptsContents)[:,:3]
    
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
        self.vertices = np.dot(self.vertices,s)

        return self.vertices

    def setVertices(self,vertices):
        self.vertices = vertices
        self.numPoints = len(vertices)

    def saveAs(self,path):
        with open(path,'w') as handle:
            for vert in self.vertices[:-1]:
                if len(vert)==3:
                    handle.write(str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\r\n")
                elif len(vert)==2:
                    handle.write(str(vert[0]) + " " + str(vert[1]) + " " + "0" + "\r\n")
            vert = self.vertices[-1]
            if len(vert)==3:
                handle.write(str(vert[0]) + " " + str(vert[1]) + " " + str(vert[2]) + "\r\n")
            elif len(vert)==2:
                handle.write(str(vert[0]) + " " + str(vert[1]) + " " + "0" + "\r\n")

    def orderByX(self):
        srcPoints = np.array(self.vertices)
        indices = np.argsort(srcPoints,0)[:,0]
        newPoints = srcPoints[indices,:]
        self.setVertices(newPoints)
