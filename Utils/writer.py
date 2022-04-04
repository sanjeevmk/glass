import csv
import numpy as np
import os

def writePoints(points,ptsFileName):
    csv.writer(open(ptsFileName,'w'),delimiter=' ').writerows(points)

def batchWritePoints(batchPoints,outputDir):
    for i in range(batchPoints.shape[0]):
        writePoints(batchPoints[i,:,:],os.path.join(outputDir,str(i)+".pts"))