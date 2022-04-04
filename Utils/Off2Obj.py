import sys
sys.path.append("../")
import os
from Utils import Obj,Off

srcDir = sys.argv[1]
dstDir = sys.argv[2]

files = os.listdir(srcDir)
offFiles = list(filter(lambda x:".off" in x,files))


for fil in offFiles:
    offInstance = Off.Off(os.path.join(srcDir,fil))
    points,faces = offInstance.readPointsAndFaces()
    objInstance = Obj.Obj("dummy.obj")
    objInstance.setVertices(points)
    objInstance.setFaces(faces)
    objInstance.saveAs(os.path.join(dstDir,fil.split(".")[0]+".obj"))
