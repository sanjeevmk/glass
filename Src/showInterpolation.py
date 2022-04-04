import os
import sys
import numpy as np

srcDir = sys.argv[1]
output = sys.argv[2]
interval = 1
files = os.listdir(srcDir)
files = list(filter(lambda x:".jpg" in x,files))
numImages = 20 #len(files)
prefix = "_".join(files[0].split(".jpg")[0].split("_")[:-1])

files = [prefix+"_"+str(i)+".jpg" for i in range(0,numImages,interval)]
inds = range(0,numImages,interval)
if inds[-1]!=numImages-1:
    files += [prefix+"_"+str(numImages-1)+".jpg"]
print(files)
from PIL import Image

imageList = []
for f in files:
    im = np.asarray(Image.open(os.path.join(srcDir,f)))
    imageList.append(im)

imageList = np.concatenate(imageList,1)
im = Image.fromarray(imageList)
im.save(output)
