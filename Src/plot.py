import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import header
import init
import sys
import train
import randomSample
import latentSpaceExplore_VanillaHMC,latentSpaceExplore_NUTSHmc,latentSpaceExplore_PerturbOptimize
from train import trainVAE

import csv 
import matplotlib.pyplot as plt
import os
import numpy as np
import json
config = sys.argv[1]
jsonArgs = json.loads(open(config,'r').read())

oprows = list(csv.reader(open(jsonArgs['plotfile'],'r'),delimiter=',',quoting=csv.QUOTE_NONNUMERIC))
#opirows = list(csv.reader(open(jsonArgs['OpiPlotFile'],'r'),delimiter=',',quoting=csv.QUOTE_NONNUMERIC))
#hmcrows = list(csv.reader(open(jsonArgs['HmcPlotFile'],'r'),delimiter=',',quoting=csv.QUOTE_NONNUMERIC))

oprows = np.squeeze(np.array(oprows))
#opirows = np.squeeze(np.array(opirows))
#hmcrows = np.squeeze(np.array(hmcrows))

opx = range(len(oprows))
#opix = range(len(opirows))
#hmcx = range(len(hmcrows))

opy = oprows
#opiy = opirows
#hmcy = hmcrows

plt.plot(opx,opy)
plt.xlabel("Number of Rounds")
plt.ylabel("Energy Value")
#plt.plot(opix,opiy)
#plt.plot(hmcx,hmcy)
plt.legend(["Pairwise Distance Energy (of latent space)"]) #,"Perturb-Optimize-Input","NUTS-HMC"])
plt.tight_layout()
if not os.path.exists("/".join(jsonArgs['plotTitle'].split("/")[:-1])):
    os.makedirs("/".join(jsonArgs['plotTitle'].split("/")[:-1]))
plt.savefig(jsonArgs['plotTitle'])
