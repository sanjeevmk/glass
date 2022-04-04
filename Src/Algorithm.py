import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import header
import init
import sys
import train_parental,train_gradients,train_regarap
import randomSample
import latentSpaceExplore_VanillaHMC,latentSpaceExplore_NUTSHmc
import latentSpaceExplore_Random,latentSpaceExplore_Tangent,latentSpaceExplore_Hessian,latentSpaceExplore_HessianClosedForm,latentSpaceExplore_HessianTopK
config = sys.argv[1]
training_params,data_classes,network_params,misc_variables,losses = init.initialize(config)
from train_parental import trainVAE
from torch.autograd import Variable
scale = Variable(torch.tensor(0.01)).cuda()
cpu_rng_state = torch.get_rng_state()
gpu_rng_state = torch.cuda.get_rng_state()
for rounds in range(training_params.startround,training_params.roundEpochs):
    if len(data_classes.expanded) > 2000 and len(data_classes.original)<=25:
        print("GLASS finished, generated {} samples".format(len(data_classes.expanded)))
        break

    if rounds>0:
        network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds-1)))

    #network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds)))
    #train_parental.trainVAE(training_params,data_classes,network_params,losses,misc_variables,rounds,scale)
    status = train_regarap.trainVAE(training_params,data_classes,network_params,losses,misc_variables,rounds,scale)

    '''
    if isinstance(status,str):
        if status=='eig failed':
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(gpu_rng_state)
            network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds-1)))
            status = train_regarap.trainVAE(training_params,data_classes,network_params,losses,misc_variables,rounds,scale)
            if isinstance(status,str):
                if status=='eig failed':
                    print("Failed second time,exit")
                    exit()
        if status=='nan error':
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(gpu_rng_state)
            network_params.autoEncoder.load_state_dict(torch.load(network_params.weightPath+"_r"+str(rounds-1)))
            status = train_regarap.trainVAE(training_params,data_classes,network_params,losses,misc_variables,rounds,scale)
    '''

    if training_params.method == 'NoEnergy':
        break
    if training_params.method == 'VanillaHMC':
        #latentSpaceExplore_VanillaHMC_Batch.hmcExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
        latentSpaceExplore_VanillaHMC.hmcExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
    elif training_params.method == 'Tangent':
        #latentSpaceExplore_Tangent.tangentExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
        #latentSpaceExplore_Hessian.hessianExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
        status = latentSpaceExplore_HessianClosedForm.hessianExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
        if isinstance(status,str):
            if status=='eig failed':
                torch.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state(gpu_rng_state)
                continue
    elif training_params.method == 'TopK':
        status = latentSpaceExplore_HessianTopK.hessianExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
        if isinstance(status,str):
            if status=='eig failed':
                torch.set_rng_state(cpu_rng_state)
                torch.cuda.set_rng_state(gpu_rng_state)
                continue
    elif training_params.method == 'Random':
        latentSpaceExplore_Random.randomExplore(training_params,data_classes,network_params,losses,misc_variables,rounds)
    else:
        break
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
