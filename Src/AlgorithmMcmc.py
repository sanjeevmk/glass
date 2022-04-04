import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import header
import init
import sys
import train_parental,train_regarap
import randomSample
import latentSpaceExplore_VanillaHMC,latentSpaceExplore_NUTSHmc
import latentSpaceExplore_Random,latentSpaceExplore_Tangent,latentSpaceExplore_Hessian
config = sys.argv[1]
training_params,data_classes,network_params,misc_variables,losses = init.initialize(config)
from train_parental import trainVAE
from torch.autograd import Variable
scale = Variable(torch.tensor(0.01)).cuda()
for rounds in range(training_params.startround,training_params.roundEpochs):
    if rounds>0:
        network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds-1)))
    train_regarap.trainVAE(training_params,data_classes,network_params,losses,misc_variables,rounds,scale)
    #exit()
    if training_params.method == 'NoEnergy':
        break
    if training_params.method == 'VanillaHMC':
        #latentSpaceExplore_VanillaHMC_Batch.hmcExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
        latentSpaceExplore_VanillaHMC.hmcExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
    elif training_params.method == 'Tangent':
        #latentSpaceExplore_Tangent.tangentExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
        latentSpaceExplore_VanillaHMC.hmcExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
        #latentSpaceExplore_Hessian.hessianExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
    elif training_params.method == 'Random':
        latentSpaceExplore_Random.randomExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
    else:
        break
    exit()
    #network_params.autoEncoder.load_state_dict(torch.load(network_params.resetWeights))
    network_params.autoEncoder.train()
    datasetLoader = header.ScapeObjMesh(data_classes.original_root,additionalDirs=[data_classes.expanded_root])
    testDatasetLoader = header.ScapeObjPoints(data_classes.original_root,additionalDirs=[data_classes.expanded_root])
    dataloader = header.DataLoader(datasetLoader,batch_size=training_params.batch,shuffle=False,num_workers=4)
    testdataloader = header.DataLoader(testDatasetLoader,batch_size=training_params.batch,shuffle=False,num_workers=2)

    data_classes.expanded = datasetLoader
    data_classes.test = testDatasetLoader
    data_classes.torch_expanded = dataloader
    data_classes.torch_test = testdataloader

    optimizer = torch.optim.Adam(network_params.autoEncoder.parameters(),lr=1e-3,weight_decay=0 ) #1e-4)
    network_params.optimizer = optimizer

    misc_variables.bestError=1e9
    misc_variables.testBestError=1e9
    training_params,data_classes,network_params,misc_variables,losses = init.initialize(config,rounds=rounds)
    #scale/=10.0
