import numpy as np 
import csv
import sys

name_f = sys.argv[1]
code_f = sys.argv[2]

codes = np.load(code_f)

names = list(csv.reader(open(name_f,'r'),delimiter=' '))
arapE = list(csv.reader(open('arap_energies.csv','r'),delimiter=' ',quoting=csv.QUOTE_NONNUMERIC))
noneE = list(csv.reader(open('none_energies.csv','r'),delimiter=' ',quoting=csv.QUOTE_NONNUMERIC))

allE = arapE + noneE
allE = np.squeeze(np.array(allE))
allE = (allE-np.min(allE))/(np.max(allE)-np.min(allE))
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

ylorbr = cm.get_cmap('YlOrBr', 8)
colors = ylorbr(100*allE)

from sklearn.manifold import TSNE

tsne = TSNE(perplexity=30,verbose=1)

embedded = tsne.fit_transform(codes)
#embedded = np.random.random((10000,2))
embedded *= 10
xmin = np.min(embedded,0)[0]
ymin = np.min(embedded,0)[1]

print(np.min(embedded,0),np.max(embedded,0))
print(xmin,ymin)
embedded[:,0] += (10-xmin)
embedded[:,1] += (10-ymin)
print(np.min(embedded,0),np.max(embedded,0))
#embedded = 100*np.random.random((10000,2))

width = int(np.max(embedded[:,0]))
height = int(np.max(embedded[:,1]))
image = 255*np.ones((height+10,width+10,3))
print(image.shape)

width += 100
height += 100
import cv2
for i in range(embedded.shape[0]):
    name = names[i]
    if 'original' in name[0]:
        color = (0,255,0)
    else:
        color = [255*x for x in colors[i][:3]]
        color = [int(c) for c in color]
        color = tuple(color)

    #cv2.circle(image,(int(embedded[i][0]),int(embedded[i][1])),2,color,-1)
    if 'original' in name[0]:
        cv2.putText(image,name[0].split('_original')[0].split('_')[-1],(int(embedded[i][0]),int(embedded[i][1])),0,0.5,0)
    cv2.circle(image,(int(embedded[i][0]),int(embedded[i][1])),6,color,-1)
    #image[int(embedded[i][1]),int(embedded[i][0]),:] = np.array(list(color))

for i in range(embedded.shape[0]):
    name = names[i]
    if 'original' in name[0]:
        cv2.putText(image,name[0].split('_original')[0].split('_')[-1],(int(embedded[i][0]),int(embedded[i][1])),0,0.5,0,2)

from PIL import Image

print(image.shape)
print(image)
im = Image.fromarray(image.astype('uint8'))
im.save("noenergy_embedding.png")
