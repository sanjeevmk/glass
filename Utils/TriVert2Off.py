import sys
import os
import csv
import numpy as np
if __name__ == "__main__":
    dataDir = sys.argv[1]
    fileNames = os.listdir(dataDir)
    vertFileNames = list(filter(lambda x:x.endswith(".vert"),fileNames))

    for vertName in vertFileNames:
        triName = vertName.split(".vert")[0]+".tri"
        vertFullName = os.path.join(dataDir,vertName)
        triFullName = os.path.join(dataDir,triName)
        verts = np.array(list(csv.reader(open(vertFullName,'r'),delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)))
        tris = np.array(list(csv.reader(open(triFullName,'r'),delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)))
        tris -= 1
        numFaceCol = np.ones((tris.shape[0],1))
        numFaceCol *= 3
        tris = np.hstack([numFaceCol,tris])
        offName = os.path.join(dataDir,vertName.split(".vert")[0]+".off")
        fh = open(offName,'w')
        fh.write("OFF\r\n")
        fh.write(str(verts.shape[0])+" "+str(tris.shape[0])+" 0\r\n")
        fh.close()
        verts= verts.tolist()
        csv.writer(open(offName,"a+"),delimiter=' ').writerows(verts)
        tris = tris.astype("int")
        tris = tris.tolist()
        csv.writer(open(offName,"a+"),delimiter=' ').writerows(tris)
