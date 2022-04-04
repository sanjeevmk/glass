import numpy as np 
import csv
import sys
import pickle
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib

data_file = sys.argv[1]
output_image = sys.argv[2]

data_dict = pickle.load(open(data_file,'rb'))

codes = []
names = []
landmark_names = ['tr_reg_000','tr_reg_001','tr_reg_002','tr_reg_003','tr_reg_004','tr_reg_005','tr_reg_006','tr_reg_007','tr_reg_008','tr_reg_009']
for k in data_dict:
    print(k)
    names.append(k)
    codes.append(data_dict[k])
exit()
from sklearn.manifold import TSNE

#tsne = TSNE(perplexity=300,verbose=1,random_state=0,n_iter=10000,learning_rate=10,early_exaggeration=2,init='pca')
tsne = TSNE(perplexity=30,verbose=1,random_state=0,n_iter=1000,learning_rate='auto',early_exaggeration=2,n_iter_without_progress=30,init='pca')

codes = np.array(codes)
embedded = tsne.fit_transform(codes)
embedded *= 14 
xmin = np.min(embedded,0)[0]
ymin = np.min(embedded,0)[1]

embedded[:,0] += (10-xmin)
embedded[:,1] += (10-ymin)

width = int(np.max(embedded[:,0]))
height = int(np.max(embedded[:,1]))
bigger = max(width,height)
image = 255*np.ones((bigger,bigger,3))

print(image.shape)
width += 100
height += 100
import cv2

ylorbr = cm.get_cmap('seismic', len(landmark_names))
colors = ylorbr(np.arange(0,1,1.0/len(landmark_names)))
colors *= 255
for i in range(embedded.shape[0]):
    for ln in landmark_names:
        if ln in names[i]:
            cv2.circle(image,(int(embedded[i][0]),int(embedded[i][1])),10,colors[landmark_names.index(ln)][:3],-1)
            break

from PIL import Image

im = Image.fromarray(image.astype('uint8'))
im.save(output_image)
