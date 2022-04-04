import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import header
import init
import sys
import train_parental
import randomSample
import latentSpaceExplore_VanillaHMC,latentSpaceExplore_NUTSHmc,InputPairInterpolation
import latentSpaceExplore_Random,latentSpaceExplore_Tangent
config = sys.argv[1]
training_params,data_classes,network_params,misc_variables,losses = init.initialize(config)
from train_parental import trainVAE
from torch.autograd import Variable
scale = Variable(torch.tensor(0.01)).cuda()
InputPairInterpolation.pairInterpolate(training_params,data_classes,network_params,losses,misc_variables,0)