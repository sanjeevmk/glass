import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import header
import init
import sys
import train_correspondence
import randomSample
import latentSpaceExplore_VanillaHMC,latentSpaceExplore_NUTSHmc
config = sys.argv[1]
training_params,data_classes,network_params,misc_variables,losses = init.initialize(config)
from train_parental import trainVAE
from torch.autograd import Variable
scale = Variable(torch.tensor(0.01)).cuda()
train_correspondence.trainVAE(training_params,data_classes,network_params,losses,misc_variables,training_params.target_round,scale)
