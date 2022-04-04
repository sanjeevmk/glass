import sys
import os

if __name__ == "__main__":
    theaSampleBinPath = sys.argv[1]
    dataDir = sys.argv[2]
    ext = "."+sys.argv[3]
    numPoints = sys.argv[4]
    uniformity = sys.argv[5]

    fileNames = os.listdir(dataDir)
    vertFileNames = list(filter(lambda x:x.endswith(ext),fileNames))

    for vertName in vertFileNames:
        command = theaSampleBinPath
        command += " -n"+numPoints
        command += " -s"+uniformity
        command += " -i "

        vertPath = os.path.join(dataDir,vertName)
        ptsPath = os.path.join(dataDir,vertName.split(ext)[0]+".pts")

        command += " " + vertPath 
        command += " dummy.obj "
        command += " " + ptsPath

        os.system(command)