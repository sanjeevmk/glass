import sys
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib
import cv2
import csv

landmark_names = ['tr_reg_000','tr_reg_001','tr_reg_002','tr_reg_003','tr_reg_004','tr_reg_005','tr_reg_006','tr_reg_007','tr_reg_008','tr_reg_009']

code_npy = sys.argv[1]
names = sys.argv[2]
#filter_names = sys.argv[3] ###
#distances = "distances.csv"

#distance_matrix = list(csv.reader(open('distances.csv','r')))
#distance_matrix = np.array(distance_matrix)

table = []
#filtered_table = []
first_row = ['filename'] + ['l' + str(i) for i in range(1,33)] + ['embed_x','embed_y']
table.append(first_row)
#filtered_table.append(first_row)

codes = np.load(code_npy)
names = open(names,'r').read().splitlines()
#filter_names = open(filter_names,'r').read().splitlines()

for i in range(len(codes)):
    row = [names[i]] + [str(x) for x in codes[i,:]] + ['0.0','0.0']
    table.append(row)

from sklearn.manifold import TSNE

tsne = TSNE(perplexity=160,verbose=1,random_state=0,n_iter=300,learning_rate=10,early_exaggeration=2,init='pca')
#custom_tsne = TSNE(perplexity=30,verbose=1,random_state=0,n_iter=500,learning_rate=10,early_exaggeration=2,metric='precomputed')
#from MulticoreTSNE import MulticoreTSNE as TSNE

codes = np.array(codes)
'''
fil_codes = []
fil_names = []
for i in range(len(codes)):
    if names[i].startswith('tr'):
        fil_codes.append(codes[i])
        fil_names.append(names[i])

codes = np.array(fil_codes)
names = list(fil_names)
'''

embedded = tsne.fit_transform(codes)
embedded *= 15

#filtered_embedded = custom_tsne.fit_transform(distance_matrix)
#filtered_embedded *= 15

xmin = np.min(embedded,0)[0]
ymin = np.min(embedded,0)[1]

embedded[:,0] += (15-xmin)
embedded[:,1] += (15-ymin)

#fxmin = np.min(filtered_embedded,0)[0]
#fymin = np.min(filtered_embedded,0)[1]

#filtered_embedded[:,0] += (10-fxmin)
#filtered_embedded[:,1] += (10-fymin)
'''
for i in range(1,len(table)):
    table[i][-2] = str(embedded[i-1,0])
    table[i][-1] = str(embedded[i-1,1])
'''

#for i in range(1,len(filtered_table)):
#    filtered_table[i][-2] = str(filtered_embedded[i-1,0])
#    filtered_table[i][-1] = str(filtered_embedded[i-1,1])

import csv
csv.writer(open('embedding_faust.csv','w'),delimiter=',').writerows(table)

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
im.save('./tsne_rounds/faust10_tsne_6.jpg')
