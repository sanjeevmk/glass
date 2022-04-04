import numpy as np
import csv
import sys
import pickle
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib

data_file = sys.argv[1]
landmark_data_file = sys.argv[2]
arap_file = sys.argv[3]

codes = []

codes = np.load(data_file)
l_codes = np.load(landmark_data_file)
codes = np.vstack([codes,l_codes])

araps = np.expand_dims(np.load(arap_file),-1)

#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

#tsne = TSNE(perplexity=300,verbose=1,random_state=0,n_iter=10000,learning_rate=10,early_exaggeration=2,init='pca')

codes = np.array(codes)
#embedded = tsne.fit_transform(codes)
embedded = TSNE(n_jobs=16,verbose=1,random_state=0,n_iter=400,perplexity=2000,early_exaggeration=2).fit_transform(codes)
embedded *= 15
xmin = np.min(embedded,0)[0]
ymin = np.min(embedded,0)[1]

embedded[:,0] += (10-xmin)
embedded[:,1] += (10-ymin)

out = np.hstack([embedded[:-10,:],araps])
out = out.tolist()

csv.writer(open('./landscape/r0_random_arap_landscape_noenergy.txt','w'),delimiter=',').writerows(out)

out = embedded[-10:,:]
csv.writer(open('./landscape/r0_landmark_tsne_noenergy.txt','w'),delimiter=',').writerows(out)

width = int(np.max(embedded[:,0]))
height = int(np.max(embedded[:,1]))
bigger = max(width,height)
image = 255*np.ones((bigger,bigger,3))

print(image.shape)
width += 100
height += 100
import cv2

#ylorbr = cm.get_cmap('seismic', len(landmark_names))
for i in range(embedded.shape[0]):
    cv2.circle(image,(int(embedded[i][0]),int(embedded[i][1])),10,(int(0),int(0),int(255)),-1)
    break

from PIL import Image

im = Image.fromarray(image.astype('uint8'))
im.save('./landscape/r0_random_tsne.jpg')
