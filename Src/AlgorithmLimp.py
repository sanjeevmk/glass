import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import header
import init
import sys
import train_limp
import randomSample
import latentSpaceExplore_VanillaHMC,latentSpaceExplore_NUTSHmc
config = sys.argv[1]
training_params,data_classes,network_params,misc_variables,losses = init.initialize(config)
from torch.autograd import Variable
scale = Variable(torch.tensor(0.01)).cuda()
for rounds in range(training_params.roundEpochs):
    train_limp.trainVAE(training_params,data_classes,network_params,losses,misc_variables,rounds,scale)
    exit()
