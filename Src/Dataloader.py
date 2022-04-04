import torch
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
import torch.utils.data as data
import os
import random
import numpy as np
import sys
sys.path.append("../")
from Utils import Off,Pts,Obj,rect_remesh
from Utils.misc import rotateX
import re
import trimesh
import time
import pointcloud_processor
class FAUSTSimple(data.Dataset):
    def __init__(self, train,data_path,target_data_path,npoints=6890, regular_sampling=True):
        self.train = train
        self.regular_sampling = regular_sampling  # sample points uniformly or proportionaly to their adjacent area
        self.npoints = npoints

        ply_data_files = os.listdir(data_path)
        data_files = os.listdir(data_path)
        data_files = [os.path.join(data_path,x) for x in data_files]
        data_files = [trimesh.load(x) for x in data_files]

        self.mesh = trimesh.load("../Data/template.ply", process=False)
        self.prop = pointcloud_processor.get_vertex_normalised_area(self.mesh)
        assert (np.abs(np.sum(self.prop) - 1) < 0.001), "Propabilities do not sum to 1!)"

        self.datas = []
        for f in data_files:
            verts = np.array(f.vertices)
            self.datas.append(torch.from_numpy(verts).float())

        self.objFiles = []
        self.objInstances = []
        self.dirobjFiles = [f.split(".ply")[0]+".obj" for f in ply_data_files]
        self.dirfiles = [os.path.join(target_data_path,f) for f in self.dirobjFiles]
        self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]

        for objInstance in self.dirobjInstances:
            objInstance.readContents()
            objInstance.computeArapMetadata()
        self.objFiles.extend(self.dirobjFiles)
        self.objInstances.extend(self.dirobjInstances)
        self.random_sample = np.random.choice(6890, size=self.npoints, p=self.prop)


    def __len__(self):
        return len(self.objInstances)

    def __getitem__(self, index):
        # LOAD a training sample

        points = self.datas[index].squeeze()
        # Clone it to keep the cached data safe
        points = points.clone()

        if self.regular_sampling:
            points = points[self.random_sample]

        target_points,_ = self.objInstances[index].readPointsAndFaces(normed=False)
        return points, target_points,self.objFiles[index]

class FAUSTObj(data.Dataset):
    def __init__(self, train,target_data_path,npoints=6890, regular_sampling=True):
        self.train = train
        self.regular_sampling = regular_sampling  # sample points uniformly or proportionaly to their adjacent area
        self.npoints = npoints

        data_files = os.listdir(target_data_path)

        self.objFiles = []
        self.objInstances = []
        self.ptsInstances = []
        self.dirobjFiles = [f for f in data_files if f.endswith(".obj")]
        self.dirfiles = [os.path.join(target_data_path,f) for f in self.dirobjFiles]
        self.dirptsfiles = [os.path.join(target_data_path,f.split(".obj")[0]+".pts") for f in self.dirobjFiles]
        self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]
        self.dirptsInstances = [Pts.Pts(path) for path in self.dirptsfiles]

        for objInstance in self.dirobjInstances:
            objInstance.readContents()
            #objInstance.computeArapMetadata()
        self.objFiles.extend(self.dirobjFiles)
        self.objInstances.extend(self.dirobjInstances)
        self.ptsInstances.extend(self.dirptsInstances)


    def __len__(self):
        return len(self.objInstances)

    def __getitem__(self, index):
        # LOAD a training sample

        target_points,_ = self.objInstances[index].readPointsAndFaces(normed=True)
        sampled_points = self.objInstances[index].samplePoints(self.npoints)
        #points,_ = self.objInstances[index].readPointsAndFaces(normed=False)
        #points = self.ptsInstances[index].readPoints()
        return sampled_points, target_points,self.objFiles[index]

class ToscaPoints(data.Dataset):
    EXT=".pts"
    def __init__(self,root,categories=[]):
        files = os.listdir(root)
        ptsFiles = list(filter(lambda x:x.endswith(ToscaPts.EXT),files))
        random.shuffle(ptsFiles)
        categoryPtsFiles = [pF for pF in ptsFiles for c in categories if c in pF]
        #categoryPtsFiles = ptsFiles
        self.categoryIds = [categories.index(c) for pF in ptsFiles for c in categories if c in pF]
        self.numClasses = len(categories)
        self.files = [os.path.join(root,f) for f in categoryPtsFiles]
        self.ptsInstances = [Pts.Pts(fileName) for fileName in self.files]

    def __len__(self):
        return len(self.ptsInstances)

    def __getitem__(self,idx):
        points = self.ptsInstances[idx].readPoints()
        catId = self.categoryIds[idx]
        label = np.zeros((self.numClasses,))
        label[catId] = 1.0
        return points,label

class ToscaPts(data.Dataset):
    EXT=".pts"
    def __init__(self,root,reader,categories=[],categoryFmt='{0:03b}',catSize=1,numClasses=2):
        files = os.listdir(root)
        ptsFiles = list(filter(lambda x:x.endswith(ToscaPts.EXT),files))
        random.shuffle(ptsFiles)
        categoryPtsFiles = [pF for pF in ptsFiles for c in categories if c in pF]
        self.categoryIds = [categories.index(c) for pF in ptsFiles for c in categories if c in pF]
        self.files = [os.path.join(root,f) for f in categoryPtsFiles]
        self.reader = reader
        self.categories = categories
        self.fmt = categoryFmt
        self.catSize = catSize
        self.numClasses = len(categories) 

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        points = self.reader(self.files[idx])
        catId = self.categoryIds[idx]
        catString = self.fmt.format(catId)
        catVector = np.array([[float(int(b)) for b in catString]])
        catFrame = np.repeat(catVector,self.catSize,axis=0)
        #catFrame = catFrame.flatten()
        label = np.zeros((self.numClasses,))
        label[catId] = 1.0
        return points,catFrame,catId,label,self.files[idx].split("/")[-1].split(".")[0]


class ToscaPtsByCategory(ToscaPts):
    def __init__(self,root,reader,categories=[],categoryFmt='{0:03b}',catSize=1):
        super(ToscaPtsByCategory,self).__init__(root,reader,categories=categories,categoryFmt=categoryFmt,catSize=catSize)

    def __len__(self):
        return len(self.categories)

    def __getitem__(self,catId):
        if catId>=len(self):
            raise IndexError("Category Id out of bounds")
        idxChoices = [idx for idx,cid in enumerate(self.categoryIds) if cid==catId]
        selectedIdx = random.choice(idxChoices)
        points = self.reader(self.files[selectedIdx])
        catString = self.fmt.format(catId)
        catVector = np.array([[float(int(b)) for b in catString]])
        catFrame = np.repeat(catVector,self.catSize,axis=0)
        #catFrame = catFrame.flatten()
        return points,catFrame,self.files[selectedIdx].split("/")[-1].split(".")[0]

    def getIds(self,catIds):
        points,catFrame,compName = self[catIds[0]]
        points = np.expand_dims(points,0)
        catFrame = np.expand_dims(catFrame,0)
        compNames = [compName]
        for cId in catIds[1:]:
            ps,cf,cn = self[cId]
            ps = np.expand_dims(ps,0)
            cf = np.expand_dims(cf,0)
            points = np.concatenate([points,ps],0)
            catFrame = np.concatenate([catFrame,cf],0)
            compNames.extend([cn])

        return points,catFrame,compNames

class ScapeOffPoints(data.Dataset):
    EXT=".off"
    cap = 10
    def __init__(self,root):
        files = os.listdir(root)
        offFiles = list(filter(lambda x:x.endswith(ScapeOffPoints.EXT),files))
        self.offFiles = offFiles[:ScapeOffPoints.cap]
        #random.shuffle(offFiles)
        self.files = [os.path.join(root,f) for f in self.offFiles]
        self.offInstances = [Off.Off(path) for path in self.files]

    def __len__(self):
        return len(self.offInstances)

    def __getitem__(self,idx):
        points,faces = self.offInstances[idx].readPointsAndFaces()
        return points,faces,self.offFiles[idx]

class ScapeOffMesh(data.Dataset):
    EXT=".off"
    cap = 10 
    def __init__(self,root):
        super(ScapeOffMesh,self).__init__()
        files = os.listdir(root)
        offFiles = list(filter(lambda x:x.endswith(ScapeOffPoints.EXT),files))
        self.offFiles = offFiles[:ScapeOffMesh.cap]
        #random.shuffle(offFiles)
        self.files = [os.path.join(root,f) for f in self.offFiles]
        self.offInstances = [Off.Off(path) for path in self.files]

        for offInstance in self.offInstances:
            offInstance.computeArapMetadata()

    def __len__(self):
        return len(self.offInstances)

    def __getitem__(self,idx):
        points,faces = self.offInstances[idx].readPointsAndFaces()
        numNeighbors,neighborsMatrix,weightMatrix = self.offInstances[idx].getArapMetadata()
        return points,faces,numNeighbors,neighborsMatrix,weightMatrix,self.offFiles[idx]

class ScapeObjPointsRotate(data.Dataset):
    EXT=".obj"
    cap = 10 
    rotations = [0.0,90.0,180.0,270.0]
    def __init__(self,root):
        super(ScapeObjPointsRotate,self).__init__()
        files = os.listdir(root)
        self.objFiles = list(filter(lambda x:x.endswith(ScapeObjPoints.EXT),files))
        #self.objFiles = objFiles[:ScapeObjMesh.cap]
        #random.shuffle(offFiles)
        self.files = [os.path.join(root,f) for f in self.objFiles]
        self.objInstances = [Obj.Obj(path) for path in self.files]

        for objInstance in self.objInstances:
            objInstance.readContents()

    def __len__(self):
        return len(self.objInstances)*len(ScapeObjPointsRotate.rotations)

    def __getitem__(self,idx):
        shapeIdx = idx//len(ScapeObjPointsRotate.rotations)
        points,faces = self.objInstances[shapeIdx].readPointsAndFaces(normed=True)
        rotIdx = idx%len(ScapeObjPointsRotate.rotations)
        angle = ScapeObjPointsRotate.rotations[rotIdx]
        points = rotateX(points,angle)
        return points,faces,self.objFiles[shapeIdx]

class ScapeObjPointsEigen(data.Dataset):
    EXT=".obj"
    cap = 10 
    def __init__(self,root,additionalDirs=[]):
        super(ScapeObjPointsEigen,self).__init__()
        self.objInstances = []
        self.objFiles = []
        for direct in [root]+additionalDirs:
            files = os.listdir(direct)
            self.dirobjFiles = list(filter(lambda x:x.endswith(ScapeObjPoints.EXT),files))
            #self.objFiles = self.objFiles[:ScapeObjMesh.cap]
            #random.shuffle(offFiles)
            self.dirfiles = [os.path.join(direct,f) for f in self.dirobjFiles]
            self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]

            for objInstance in self.dirobjInstances:
                objInstance.readContents()
                objInstance.computeArapMetadata()
            self.objInstances.extend(self.dirobjInstances)
            self.objFiles.extend(self.dirobjFiles)

    def __len__(self):
        return len(self.objInstances)

    def __getitem__(self,idx):
        points,faces = self.objInstances[idx].readPointsAndFaces(normed=True)
        eigC,eigV = self.objInstances[idx].getEigens()
        return points,faces,eigC,self.objFiles[idx]

class ScapeObjPoints(data.Dataset):
    EXT=".obj"
    cap = 10 
    def __init__(self,root,additionalDirs=[]):
        super(ScapeObjPoints,self).__init__()
        self.objInstances = []
        self.objFiles = []
        iterin=0
        for direct in [root]+additionalDirs:
            files = os.listdir(direct)
            self.dirobjFiles = list(filter(lambda x:x.endswith(ScapeObjPoints.EXT),files))
            #self.objFiles = self.objFiles[:ScapeObjMesh.cap]
            #random.shuffle(offFiles)
            self.dirfiles = [os.path.join(direct,f) for f in self.dirobjFiles]
            self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]

            for objInstance in self.dirobjInstances:
                objInstance.readContents()
            self.objInstances.extend(self.dirobjInstances)
            self.objFiles.extend(self.dirobjFiles)
            iterin+=1
            if iterin==1:
                break

    def __len__(self):
        return len(self.objInstances)

    def __getitem__(self,idx):
        points,faces = self.objInstances[idx].readPointsAndFaces(normed=False)
        #corner = self.objInstances[idx].getCorner()
        #newPoints,newFaces,corner = rect_remesh.rectRemesh(points,faces,corner_vid=corner)
        #self.objInstances[idx].setVertices(newPoints)
        #self.objInstances[idx].setFaces(newFaces)
        #self.objInstances[idx].setCorner(corner)

        #points,faces = self.objInstances[idx].readPointsAndFaces(normed=False)
        #corner = self.objInstances[idx].getCorner()

        return points,faces,self.objFiles[idx]

class ScapeObjMeshPairs(data.Dataset):
    EXT=".obj"
    cap = 10 
    def __init__(self,root,additionalDirs = []):
        super(ScapeObjMeshPairs,self).__init__()
        self.objFiles = []
        objInstances = []
        self.pairObjInstances = []
        for direct in [root]+additionalDirs:
            files = os.listdir(direct)
            self.dirobjFiles = list(filter(lambda x:x.endswith(ScapeObjMesh.EXT),files))
            print(len(self.dirobjFiles))
            self.dirfiles = [os.path.join(direct,f) for f in self.dirobjFiles]
            self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]

            for objInstance in self.dirobjInstances:
                objInstance.readContents()
                objInstance.computeArapMetadata()
            self.objFiles.extend(self.dirobjFiles)
            objInstances.extend(self.dirobjInstances)
        pairIndices = []
        for i in range(len(self.objFiles)):
            for j in range(len(self.objFiles)):
                if i==j:
                    continue
                if (j,i) in pairIndices:
                    continue
                pairIndices.append((i,j))
        self.pairObjInstances = [(objInstances[i],objInstances[j]) for i,j in pairIndices]
        self.pairObjFiles = [(self.objFiles[i],self.objFiles[j]) for i,j in pairIndices]

    def __len__(self):
        return len(self.pairObjInstances)

    def __getitem__(self,idx):
        points0,faces0 = self.pairObjInstances[idx][0].readPointsAndFaces(normed=True)
        numNeighbors0,accnumneighbors0,neighborsMatrix0,weightMatrix0,_ = self.pairObjInstances[idx][0].getArapMetadata()
        area0 = self.pairObjInstances[idx][0].getArea()
        area0 = np.array([area0])

        points1,faces1 = self.pairObjInstances[idx][1].readPointsAndFaces(normed=True)
        numNeighbors1,accnumneighbors1,neighborsMatrix1,weightMatrix1,_ = self.pairObjInstances[idx][1].getArapMetadata()
        area1 = self.pairObjInstances[idx][1].getArea()
        area1 = np.array([area1])
        return points0,faces0,numNeighbors0,accnumneighbors0,neighborsMatrix0,weightMatrix0,area0,self.pairObjFiles[idx][0],points1,faces1,numNeighbors1,accnumneighbors1,neighborsMatrix1,weightMatrix1,area1,self.pairObjFiles[idx][1]

class ScapeObjMeshEigen(data.Dataset):
    EXT=".obj"
    cap = 10 
    def __init__(self,root,additionalDirs = []):
        super(ScapeObjMeshEigen,self).__init__()
        self.objFiles = []
        self.objInstances = []
        for direct in [root]+additionalDirs:
            files = os.listdir(direct)
            self.dirobjFiles = list(filter(lambda x:x.endswith(ScapeObjMesh.EXT),files))
            self.dirfiles = [os.path.join(direct,f) for f in self.dirobjFiles]
            self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]

            for objInstance in self.dirobjInstances:
                objInstance.readContents()
                objInstance.computeArapMetadata()
            self.objFiles.extend(self.dirobjFiles)
            self.objInstances.extend(self.dirobjInstances)

    def __len__(self):
        return len(self.objInstances)

    def __getitem__(self,idx):
        points,faces = self.objInstances[idx].readPointsAndFaces(normed=True)
        numNeighbors,neighborsMatrix,weightMatrix = self.objInstances[idx].getArapMetadata()
        eigC,eigV = self.objInstances[idx].getEigens()
        area = self.objInstances[idx].getArea()
        area = np.array([area])
        return points,faces,numNeighbors,neighborsMatrix,weightMatrix,area,eigC,eigV,eigV.T,self.objFiles[idx]

class ToonObjMesh(data.Dataset):
    EXT=".obj"
    cap = 10 
    def __init__(self,root,additionalDirs = []):
        super(ToonObjMesh,self).__init__()
        self.objFiles = []
        self.objInstances = []
        self.classes = []
        self.classmap = {}
        for direct in [root]:
            files = os.listdir(direct)
            self.dirobjFiles = list(filter(lambda x:x.endswith(ScapeObjMesh.EXT),files))
            self.dirfiles = [os.path.join(direct,f) for f in self.dirobjFiles]
            self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]
            for name in self.dirobjFiles:
                pref = name.split(".obj")[0]
                classind = len(self.classmap)
                self.classmap[pref] = classind
            numClasses = len(self.classmap)
            for name in self.dirobjFiles:
                pref = name.split(".obj")[0]
                classind = self.classmap[pref]
                classv = [0.0]*numClasses
                classv[classind] = 1.0
                self.classmap[pref] = classv
                self.classes.append(classv)

            for objInstance in self.dirobjInstances:
                objInstance.readContents()
                objInstance.computeArapMetadata()
            self.objFiles.extend(self.dirobjFiles)
            self.objInstances.extend(self.dirobjInstances)

        for direct in additionalDirs:
            files = os.listdir(direct)
            self.dirobjFiles = list(filter(lambda x:x.endswith(ScapeObjMesh.EXT),files))
            self.dirfiles = [os.path.join(direct,f) for f in self.dirobjFiles]
            self.p_dirfiles = [os.path.join(direct,f.split(".")[0]+".pts") for f in self.dirobjFiles]
            self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]

            for i,objInstance in enumerate(self.dirobjInstances):
                fname = self.dirobjFiles[i]
                classname = re.search('frame_(.*)_triangle',fname).group(0)
                self.classes.append(self.classmap[classname])
                objInstance.readContents()
                objInstance.computeArapMetadata()
                objInstance.setParentPath(self.p_dirfiles[i])
            self.objFiles.extend(self.dirobjFiles)
            self.objInstances.extend(self.dirobjInstances)


    def __len__(self):
        return len(self.objInstances)

    def __getitem__(self,idx):
        points,faces = self.objInstances[idx].readPointsAndFaces(normed=False)
        '''
        corner = self.objInstances[idx].getCorner()
        if corner == -1:
            newPoints,newFaces,corner = rect_remesh.rectRemesh(points,faces,corner_vid=corner)
            self.objInstances[idx].setVertices(newPoints)
            self.objInstances[idx].setFaces(newFaces)
            self.objInstances[idx].setCorner(corner)

        points,faces = self.objInstances[idx].readPointsAndFaces(normed=False)
        corner = self.objInstances[idx].getCorner()
        '''

        numNeighbors,neighborsMatrix,weightMatrix = self.objInstances[idx].getArapMetadata()
        #p_points = self.objInstances[idx].getParentPoints()
        area = self.objInstances[idx].getArea()
        area = np.array([area])
        return points,faces,numNeighbors,neighborsMatrix,weightMatrix,area,self.objFiles[idx],points,np.array(self.classes[idx])

class Dfaust(data.Dataset):
    EXT=".obj"
    cap = 10 
    def __init__(self,root,keypoints_file):
        super(Dfaust,self).__init__()
        keypoint_contents = []
        with open(keypoints_file,'r') as f:
            keypoint_contents = f.read().splitlines()
        seq_folder = os.path.join(root,keypoint_contents[0])
        keypoint_frame_numbers = [int(x) for x in keypoint_contents[1:]]
        last_keypoint_frame = max(keypoint_frame_numbers)
        first_keypoint_frame = min(keypoint_frame_numbers)
        self.objFiles = []
        self.objInstances = []
        self.key_indices = []

        for i,frame_number in enumerate(range(first_keypoint_frame,last_keypoint_frame+1)):
            objFile = str(frame_number).zfill(5)+'.obj'
            self.objFiles.append(objFile)
            objInstance = Obj.Obj(os.path.join(seq_folder,objFile))
            objInstance.readContents()
            self.objInstances.append(objInstance)
            if frame_number in keypoint_frame_numbers:
                self.key_indices.append(i)

    def __len__(self):
        return len(self.objInstances)

    def __getitem__(self,idx):
        points,faces = self.objInstances[idx].readPointsAndFaces(normed=True)
        return points,faces,self.objFiles[idx]

class ScapeObjMeshFullArap(data.Dataset):
    EXT=".obj"
    cap = 10 
    def __init__(self,root,additionalDirs = []):
        super(ScapeObjMeshFullArap,self).__init__()
        self.objFiles = []
        self.objInstances = []
        for direct in [root]:
            files = sorted(os.listdir(direct))
            self.dirobjFiles = list(filter(lambda x:x.endswith(ScapeObjMesh.EXT),files))
            print(len(self.dirobjFiles))
            self.dirfiles = [os.path.join(direct,f) for f in self.dirobjFiles]
            self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]
            iterin=0

            for i,objInstance in enumerate(self.dirobjInstances):
                print(i,len(self.dirobjInstances))
                objInstance.readContents()
                #objInstance.computeArapMetadata()
                iterin+=1
                #if iterin==10:
                #    break
            self.objFiles.extend(self.dirobjFiles)
            self.objInstances.extend(self.dirobjInstances)

        for direct in additionalDirs:
            files = sorted(os.listdir(direct))
            self.dirobjFiles = list(filter(lambda x:x.endswith(ScapeObjMesh.EXT),files))
            print(len(self.dirobjFiles))
            self.dirfiles = [os.path.join(direct,f) for f in self.dirobjFiles]
            self.p_dirfiles = [os.path.join(direct,f.split(".")[0]+".pts") for f in self.dirobjFiles]
            self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]

            for i,objInstance in enumerate(self.dirobjInstances):
                objInstance.readContents()
                #objInstance.computeArapMetadata()
                objInstance.setParentPath(self.p_dirfiles[i])
            self.objFiles.extend(self.dirobjFiles)
            self.objInstances.extend(self.dirobjInstances)


    def __len__(self):
        return len(self.objInstances)

    def __getitem__(self,idx):
        points,faces = self.objInstances[idx].readPointsAndFaces(normed=True)
        numNeighbors,accnumneighbors,neighborsMatrix,weightMatrix,square_weightMatrix = self.objInstances[idx].getArapMetadata()
        #area = self.objInstances[idx].getArea()
        #area = np.array([area])
        #return points,faces,self.objFiles[idx]
        return points,faces,numNeighbors,accnumneighbors,neighborsMatrix,weightMatrix,self.objFiles[idx],square_weightMatrix

class ScapeObjMesh(data.Dataset):
    EXT=".obj"
    cap = 10 
    def __init__(self,root,additionalDirs = []):
        super(ScapeObjMesh,self).__init__()
        self.objFiles = []
        self.objInstances = []
        for direct in [root]:
            files = sorted(os.listdir(direct))
            self.dirobjFiles = list(filter(lambda x:x.endswith(ScapeObjMesh.EXT),files))
            print(len(self.dirobjFiles))
            self.dirfiles = [os.path.join(direct,f) for f in self.dirobjFiles]
            self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]
            iterin=0

            for i,objInstance in enumerate(self.dirobjInstances):
                print(i,len(self.dirobjInstances))
                objInstance.readContents()
                #objInstance.computeArapMetadata()
                iterin+=1
                #if iterin==10:
                #    break
            self.objFiles.extend(self.dirobjFiles)
            self.objInstances.extend(self.dirobjInstances)

        for direct in additionalDirs:
            files = sorted(os.listdir(direct))
            self.dirobjFiles = list(filter(lambda x:x.endswith(ScapeObjMesh.EXT),files))
            print(len(self.dirobjFiles))
            self.dirfiles = [os.path.join(direct,f) for f in self.dirobjFiles]
            self.p_dirfiles = [os.path.join(direct,f.split(".")[0]+".pts") for f in self.dirobjFiles]
            self.dirobjInstances = [Obj.Obj(path) for path in self.dirfiles]

            for i,objInstance in enumerate(self.dirobjInstances):
                objInstance.readContents()
                #objInstance.computeArapMetadata()
                objInstance.setParentPath(self.p_dirfiles[i])
            self.objFiles.extend(self.dirobjFiles)
            self.objInstances.extend(self.dirobjInstances)


    def __len__(self):
        return len(self.objInstances)

    def __getitem__(self,idx):
        points,faces = self.objInstances[idx].readPointsAndFaces(normed=True)
        #numNeighbors,accnumneighbors,neighborsMatrix,weightMatrix,square_weightMatrix = self.objInstances[idx].getArapMetadata()
        #area = self.objInstances[idx].getArea()
        #area = np.array([area])
        return points,faces,self.objFiles[idx]
        #return points,faces,numNeighbors,accnumneighbors,neighborsMatrix,weightMatrix,area,self.objFiles[idx],points,square_weightMatrix

class ScapeObjMeshRotate(data.Dataset):
    EXT=".obj"
    cap = 10 
    rotations = [0.0,90.0,180.0,270.0]
    def __init__(self,root):
        super(ScapeObjMeshRotate,self).__init__()
        files = os.listdir(root)
        self.objFiles = list(filter(lambda x:x.endswith(ScapeObjMeshRotate.EXT),files))
        #self.objFiles = objFiles[:ScapeObjMesh.cap]
        #random.shuffle(offFiles)
        self.files = [os.path.join(root,f) for f in self.objFiles]
        self.objInstances = [Obj.Obj(path) for path in self.files]

        for objInstance in self.objInstances:
            objInstance.readContents()
            objInstance.computeArapMetadata()

    def __len__(self):
        return len(self.objInstances)*len(ScapeObjMeshRotate.rotations)

    def __getitem__(self,idx):
        shapeIdx = idx//len(ScapeObjPointsRotate.rotations)
        points,faces = self.objInstances[shapeIdx].readPointsAndFaces(normed=True)
        rotIdx = idx%len(ScapeObjMeshRotate.rotations)
        angle = ScapeObjPointsRotate.rotations[rotIdx]
        points = rotateX(points,angle)
        numNeighbors,neighborsMatrix,weightMatrix = self.objInstances[shapeIdx].getArapMetadata()
        return points,faces,numNeighbors,neighborsMatrix,weightMatrix,self.objFiles[shapeIdx]

class GaussianData(data.Dataset):
    def __init__(self,gaussians,extra_gaussians=[]):
        super(GaussianData,self).__init__()
        self.gaussians = gaussians + extra_gaussians

    def __len__(self):
        return len(self.gaussians)

    def __getitem__(self,idx):
        mu,cov,covinv = self.gaussians[idx].get()
        return mu,cov,covinv
