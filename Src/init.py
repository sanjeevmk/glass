import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import json
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

import header
from extension.arap.cuda.arap import Arap
from extension.arap_closed.cuda.arap import ClosedArap as ArapSolveRhs
#from extension.isometric.cuda.isometric import Isometric
from extension.grad_arap.cuda.arap import ArapGrad
from extension.bending.cuda.arap import Bending
#from extension.grad_isometric.cuda.isometric import IsometricGrad
#from extension.asap.cuda.asap import Asap
#from extension.arap2d.cuda.arap import Arap as Arap2D
#from extension.asap2d.cuda.asap import Asap as Asap2D
#from extension.carap.cuda.carap import CArap
#from extension.casap.cuda.casap import CAsap
#from extension.pairwisedistance.pairwisedistance import PairwiseDistances
#from extension.chamfer.dist_chamfer import chamferDist as CD
import sys
sys.path.append("../")
from Utils import Obj

mse = nn.MSELoss()
arap = Arap()
bending = Bending()
arap_solve_rhs = ArapSolveRhs()
#isometric = Isometric()
arapgrad = ArapGrad()
#isometricgrad = IsometricGrad()
#arap2d = Arap2D()
#carap = CArap()
#casap = CAsap()
#asap = Asap()
#asap2d = Asap2D()
#cd = CD()
#pdist = PairwiseDistances().cuda()
regWeight = Variable(torch.tensor(1.0)).cuda() #,requires_grad=False)
regularizer = header.NormalReg(regWeight=regWeight)

class TrainingParams: pass 
class DataClasses: pass 
class NetworkParams: pass
class MiscVariables: pass
class Losses: pass

def initialize(config,rounds=-1):
    jsonArgs = json.loads(open(config,'r').read())
    startepochs = 0
    if 'scaperoot' in jsonArgs:
        if 'interp' in jsonArgs['scaperoot'] or 'arapsi' in jsonArgs['scaperoot']:
            startepochs = jsonArgs['laterepoch']
        else:
            startepochs = jsonArgs['startepoch']
    laterepochs = jsonArgs['laterepoch']
    correpochs = 3000
    roundEpochs = 15
    if not 'hmcEpochs' in jsonArgs:
        hmcEpochs = 3
    else:
        hmcEpochs = jsonArgs['hmcEpochs']
    samplingRounds = 1000
    optEpochs = 300 
    perturbRounds = 5
    batch = jsonArgs['batch']
    training_params = TrainingParams()
    training_params.epochs = laterepochs
    training_params.startepochs = startepochs
    training_params.correpochs = correpochs
    training_params.roundEpochs = roundEpochs
    training_params.hmcEpochs = hmcEpochs
    training_params.optEpochs = optEpochs
    training_params.samplingRounds = samplingRounds
    training_params.perturbRounds = perturbRounds
    training_params.batch = batch
    training_params.method = jsonArgs['method']
    training_params.energy = jsonArgs['energy']
    training_params.numsteps = jsonArgs['numsteps']
    training_params.stepsize = float(jsonArgs['stepsize'])
    training_params.bestenergy = float(jsonArgs['bestenergy'])
    training_params.projectsteps = jsonArgs['numprojectsteps']
    training_params.startround = jsonArgs['startRound']
    training_params.project = jsonArgs['project']
    training_params.ncomp = jsonArgs['ncomp']
    if 'norm' in jsonArgs:
        training_params.norm = jsonArgs['norm']
    else:
        training_params.norm = 2

    if 'round' in jsonArgs:
        training_params.target_round = jsonArgs['round']
    if 'test_round' in jsonArgs:
        training_params.test_round = jsonArgs['test_round']
    #if 'sampling' in jsonArgs:
    if 'limp' in jsonArgs['sampleDir'] or 'vae' in jsonArgs['sampleDir'] or 'limp' in jsonArgs['hmcsampleDir'] or 'vae' in jsonArgs['hmcsampleDir']:
        training_params.sampling = True #jsonArgs['sampling']
        training_params.num_samples = 2000
        #if jsonArgs['sampling']:
        #    training_params.num_samples = jsonArgs['num_samples']
    else:
        training_params.sampling = False

    dims=jsonArgs['dims']
    pathDir = "/".join(jsonArgs['modelPath'].split("/")[:-1])
    if 'corrroot' in jsonArgs:
        corrpathDir = "/".join(jsonArgs['corrmodelPath'].split("/")[:-1])

    if not os.path.exists(pathDir):
        os.makedirs(pathDir)
    if 'corrroot' in jsonArgs and  not os.path.exists(corrpathDir):
        os.makedirs(corrpathDir)

    if not os.path.exists(jsonArgs['sampleDir']):
        os.makedirs(jsonArgs['sampleDir'])
    if not os.path.exists(jsonArgs['reconDir']):
        os.makedirs(jsonArgs['reconDir'])
    if not os.path.exists(jsonArgs['hmcsampleDir']):
        os.makedirs(jsonArgs['hmcsampleDir'])


    if 'dfaust_root' in jsonArgs:
        originaldatasetLoader = header.Dfaust(jsonArgs['dfaust_root'],jsonArgs['dfaust_keypoints_file'])
    else:
        originaldatasetLoader = header.ScapeObjMesh(jsonArgs['scaperoot'])

    if jsonArgs['train']:
        additionalDirs = [jsonArgs['sampleDir']]
    else:
        additionalDirs = []

    if 'dfaust_root' in jsonArgs:
        datasetLoader = header.Dfaust(jsonArgs['dfaust_root'],jsonArgs['dfaust_keypoints_file'])
    else:
        datasetLoader = header.ScapeObjMesh(jsonArgs['scaperoot'],additionalDirs=additionalDirs)
        full_arap_datasetLoader = header.ScapeObjMeshFullArap(jsonArgs['scaperoot'],additionalDirs=additionalDirs)
        full_arap_original_datasetLoader = header.ScapeObjMeshFullArap(jsonArgs['scaperoot'],additionalDirs=[])

    if 'limp' in jsonArgs:
        if jsonArgs['limp']:
            pairdatasetLoader = header.ScapeObjMeshPairs(jsonArgs['scaperoot'])

    #testDatasetLoader = header.ScapeObjPoints(jsonArgs['scaperoot'])

    originaldataloader = DataLoader(originaldatasetLoader,batch_size=batch,shuffle=False,num_workers=4)
    if 'dfaust_root' not in jsonArgs:
        full_arap_dataloader = DataLoader(full_arap_datasetLoader,batch_size=batch,shuffle=False,num_workers=4)
        full_arap_original_dataloader = DataLoader(full_arap_original_datasetLoader,batch_size=batch,shuffle=False,num_workers=4)
    dataloader = DataLoader(datasetLoader,batch_size=batch,shuffle=True,num_workers=4)
    if 'limp' in jsonArgs and jsonArgs['limp']:
        pairdataloader = DataLoader(pairdatasetLoader,batch_size=batch,shuffle=True,num_workers=4)

    #testdataloader = DataLoader(testDatasetLoader,batch_size=1,shuffle=True,num_workers=4)

    if 'corrroot' in jsonArgs:
        #corrDatasetLoader = header.FAUSTSimple(jsonArgs['train'],jsonArgs['corrroot'],jsonArgs['scaperoot'],npoints=jsonArgs['corrnumPoints'])
        corrDatasetLoader = header.FAUSTObj(jsonArgs['train'],jsonArgs['corrroot'],npoints=jsonArgs['corrnumPoints'])
        corrdataloader = DataLoader(corrDatasetLoader,batch_size=batch,shuffle=False,num_workers=4)

    data_classes = DataClasses()
    data_classes.original = originaldatasetLoader 
    if 'corrroot' in jsonArgs:
        data_classes.corrdata = corrDatasetLoader
    data_classes.expanded = datasetLoader
    if 'dfaust_root' not in jsonArgs:
        data_classes.full_arap = full_arap_datasetLoader
        data_classes.full_arap_original = full_arap_original_datasetLoader
    #data_classes.test = testDatasetLoader
    data_classes.torch_original = originaldataloader
    if 'dfaust_root' not in jsonArgs:
        data_classes.torch_full_arap = full_arap_dataloader
        data_classes.torch_full_arap_original = full_arap_original_dataloader
    if 'corrroot' in jsonArgs:
        data_classes.torch_corrdata = corrdataloader
    data_classes.torch_expanded = dataloader
    if 'limp' in jsonArgs and jsonArgs['limp']:
        data_classes.torch_pair = pairdataloader
    #data_classes.torch_test = testdataloader
    if 'dfaust_root' in jsonArgs:
        data_classes.original_root = jsonArgs['dfaust_root']
    else:
        data_classes.original_root = jsonArgs['scaperoot']
    data_classes.test_set = ""

    possible_test_path = None
    if 'scaperoot' in jsonArgs:
        possible_test_path = jsonArgs['scaperoot'][:-1] + "_test/"

    data_classes.testset_original = None
    if possible_test_path is not None and os.path.exists(possible_test_path):
        data_classes.test_set = possible_test_path
        testset_full_arap_original_datasetLoader = header.ScapeObjMeshFullArap(data_classes.test_set,additionalDirs=[])
        testset_full_arap_original_dataloader = DataLoader(testset_full_arap_original_datasetLoader,batch_size=batch,shuffle=False,num_workers=4)
        data_classes.testset_torch_full_arap_original = testset_full_arap_original_dataloader
        testset_originaldatasetLoader = header.ScapeObjMesh(possible_test_path)
        data_classes.testset_original = testset_originaldatasetLoader

    data_classes.expanded_root = jsonArgs['sampleDir']

    if 'templateHigh' in jsonArgs:
        data_classes.template_hi = jsonArgs['templateHigh']
        data_classes.template_low = jsonArgs['templateLow']
        data_classes.high_folder = jsonArgs['highFolder']
        data_classes.high_prefix = [x.split('.')[0] for x in os.listdir(data_classes.high_folder)]
        _obj = Obj.Obj(data_classes.template_hi)
        _obj.readContents()
        _,faces = _obj.readPointsAndFaces(normed=True)
        data_classes.high_faces = faces

    autoEncoder = header.VertexNetAutoEncoderFC(num_points=jsonArgs['numPoints'],num_dims=dims,bottleneck=jsonArgs['bottleneck'],catSize=0,rate=1e-3,numsteps=training_params.projectsteps,energy=training_params.energy).cuda()
    autoEncoder.apply(header.weights_init)
    if 'corrroot' in jsonArgs:
        corrEncoder = header.PointNetCorr(npoint=jsonArgs['corrnumPoints'],nlatent=jsonArgs['bottleneck']).cuda()
        corrEncoder.apply(header.weights_init)
    #classifier = header.SingleLayerClassifier(jsonArgs['bottleneck'],len(originaldatasetLoader)).cuda()
    #optimizer = torch.optim.Adam(list(autoEncoder.parameters())+list(classifier.parameters()),lr=1e-3,weight_decay=0 ) #1e-4)
    optimizer = torch.optim.Adam(autoEncoder.parameters(),lr=1e-3,weight_decay=0 ) #1e-4)
    if 'corrroot' in jsonArgs:
        corroptimizer = torch.optim.Adam(corrEncoder.parameters(),lr=1e-2,weight_decay=0 ) #1e-4)
    noise = torch.FloatTensor(batch, jsonArgs['bottleneck']).requires_grad_(False).cuda()
    alpha = torch.FloatTensor(batch,1).requires_grad_(False).cuda()

    if training_params.method!="NoEnergy":
        #energyWeight = Variable(torch.tensor(1e-9)).cuda()
        energyWeight = Variable(torch.tensor(float(jsonArgs['energyweight']))).cuda()
        testEnergyWeight = Variable(torch.tensor(1.0)).cuda() #,requires_grad=False)
    else:
        energyWeight = Variable(torch.tensor(0.0)).cuda()
        testEnergyWeight = Variable(torch.tensor(1.0)).cuda() #,requires_grad=False)

    network_params = NetworkParams()
    network_params.autoEncoder = autoEncoder
    if 'corrroot' in jsonArgs:
        network_params.corrEncoder = corrEncoder
    #network_params.classifier = classifier
    network_params.optimizer = optimizer
    if 'corrroot' in jsonArgs:
        network_params.corroptimizer = corroptimizer
    network_params.weightPath = jsonArgs['modelPath']
    if 'corrroot' in jsonArgs:
        network_params.corrweightPath = jsonArgs['corrmodelPath']
    network_params.resetWeights = jsonArgs['resetWeights']
    network_params.noise = noise
    network_params.alpha = alpha
    network_params.energyWeight = energyWeight
    network_params.testEnergyWeight = testEnergyWeight
    network_params.dims = dims
    network_params.bottleneck = jsonArgs['bottleneck']
    network_params.numPoints = jsonArgs['numPoints']

    torch.save(autoEncoder.state_dict(),jsonArgs["resetWeights"])

    autoEncoder.train()

    bestError = 1e9
    testBestError = 1e9
    hmcThresh=1e-2
    alpha = jsonArgs['alpha']

    misc_variables = MiscVariables()
    misc_variables.bestError = bestError
    misc_variables.testBestError = testBestError
    misc_variables.hmcThresh = hmcThresh
    misc_variables.alpha = alpha
    misc_variables.numNewSamples = 1
    misc_variables.sampleDir = jsonArgs['sampleDir']
    misc_variables.reconDir = jsonArgs['reconDir']
    misc_variables.hmcsampleDir = jsonArgs['hmcsampleDir']
    misc_variables.outputDir = jsonArgs['outputDir']
    misc_variables.randoutputDir = jsonArgs['randoutputDir']
    if 'feature' in jsonArgs:
        misc_variables.featureFile = jsonArgs['feature']
    if 'num_shapes' in jsonArgs:
        misc_variables.num_shapes = jsonArgs['num_shapes']
    if 'corrroot' in jsonArgs:
        misc_variables.faustType = jsonArgs['faustType']
        misc_variables.test_corr_root = jsonArgs['testcorrroot']
    losses = Losses()
    losses.mse = mse
    losses.arap = arap
    losses.arap_solve_rhs = arap_solve_rhs
    losses.bending = bending
    #losses.isometric = isometric
    losses.arapgrad = arapgrad
    #losses.isometricgrad = isometricgrad
    #losses.arap2d = arap2d
    #losses.carap = carap
    #losses.asap = asap
    #losses.asap2d = asap2d
    #losses.pdist = pdist
    #losses.cd = cd
    losses.regularizer = regularizer
    #losses.cross_entropy = torch.nn.BCELoss(reduction='none')


    return training_params,data_classes,network_params,misc_variables,losses
