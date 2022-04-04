import numpy as np
import csv

def readPoints(ptsFileName):
    ptsContents = list(csv.reader(open(ptsFileName,'r'),delimiter=' ',quoting=csv.QUOTE_NONNUMERIC))
    points = np.array(ptsContents)
    bmin = np.min(points,axis=0)
    bmax = np.max(points,axis=0)
    diagsq = np.sum(np.power(bmax-bmin,2))
    diag = np.sqrt(diagsq)
    s = np.eye(3)
    s *= (1.0/diag)
    points = np.dot(points,s)
    return points