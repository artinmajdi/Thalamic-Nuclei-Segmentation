import nibabel as nib
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# import skimage
# print(skimage.__version__)
from skimage import measure
from copy import deepcopy
import json
from scipy import ndimage

# TODO: use os.path.dirname & os.path.abspath instead of '/' remover
def NucleiSelection(ind = 1):

    def func_NucleusName(ind):
        if ind in range(20):
            if ind == 1:
                NucleusName = '1-THALAMUS'
            elif ind == 2:
                NucleusName = '2-AV'
            elif ind == 4:
                NucleusName = '4-VA'
            elif ind == 5:
                NucleusName = '5-VLa'
            elif ind == 6:
                NucleusName = '6-VLP'
            elif ind == 7:
                NucleusName = '7-VPL'
            elif ind == 8:
                NucleusName = '8-Pul'
            elif ind == 9:
                NucleusName = '9-LGN'
            elif ind == 10:
                NucleusName = '10-MGN'
            elif ind == 11:
                NucleusName = '11-CM'
            elif ind == 12:
                NucleusName = '12-MD-Pf'
            elif ind == 13:
                NucleusName = '13-Hb'
            elif ind == 14:
                NucleusName = '14-MTT'
        else:
            if ind == 1.1:
                NucleusName = 'lateral_ImClosed'
            elif ind == 1.2:
                NucleusName = 'posterior_ImClosed'
            elif ind == 1.3:
                NucleusName = 'Medial_ImClosed'
            elif ind == 1.4:
                NucleusName = 'Anterior_ImClosed'
            elif ind == 1.9:
                NucleusName = 'HierarchicalCascade'

        return NucleusName

    def func_FullIndexes(ind):
        if ind in range(20):
            return [1,2,4,5,6,7,8,9,10,11,12,13,14]
        elif ind == 1.1: # lateral
            return [4,5,6,7]
        elif ind == 1.2: # posterior
            return [8,9,10]
        elif ind == 1.3: # 'Medial'
            return [11,12,13]
        elif ind == 1.4:
            return [2]
        elif ind == 1.9:
            return [1.1 , 1.2 , 1.3 , 1.4]

    name = func_NucleusName(ind)
    FullIndexes = func_FullIndexes(ind)
    Full_Names = [func_NucleusName(ix) for ix in FullIndexes]

    return name, FullIndexes, Full_Names

# TODO: repalce all NucleiSelection()  with Nuclei_Class class                       
class Nuclei_Class():        
        
    def __init__(self, index=1, method = 'HCascade'):

        def dic_Name(index):
            switcher = {
                1: '1-THALAMUS',
                2: '2-AV',
                4: '4-VA',
                5: '5-VLa',
                6: '6-VLP',
                7: '7-VPL',
                8: '8-Pul',
                9: '9-LGN',
                10: '10-MGN',
                11: '11-CM',
                12: '12-MD-Pf',
                13: '13-Hb',
                14: '14-MTT',
                1.1: 'lateral_ImClosed',
                1.2: 'posterior_ImClosed',
                1.3: 'Medial_ImClosed',
                1.4: 'Anterior_ImClosed',
                1.9: 'HierarchicalCascade' }  
            return switcher.get(index, 'wrong index')
        self.name = dic_Name(index)

        self.dic_Name = dic_Name
        self.method      = method
        self.child       = ()
        self.parent      = None
        self.grandparent = None
        self.index = index
        

        def find_Parent_child(self):
            def parent_child(index):
                if self.method == 'HCascade':
                    switcher_Parent = {
                        # 1:   (None, [1.1 , 1.2 , 1.3 , 2]),
                        1:   (None, [1.1 , 1.2 , 1.3 , 1.4]),
                        1.1: (1,    [4,5,6,7]),   # Lateral
                        1.2: (1,    [8,9,10]),    # Posterior
                        1.3: (1,    [11,12,13]),  # Medial                       
                        1.4: (1,     [2])}       # Anterior
                        # 1.4: (1,     [None]),       # Anterior                        
                        # 2:   (1,     None) }              
                    return switcher_Parent.get(index)
                else:
                    return ( None, [2,4,5,6,7,8,9,10,11,12,13,14] ) if index == 1 else (1,None)                               

            def func_HCascade(self):
                                                                                       
                if parent_child(self.index): 
                    self.parent , self.child = parent_child(self.index)
                else: 
                    for ix in parent_child(1)[1]:
                        HC_parent, HC_child = parent_child(ix)
                        if HC_child and self.index in HC_child: self.grandparent , self.parent , self.child = (HC_parent , ix , None)

            if   self.method == 'Cascade':  self.grandparent , self.parent , self.child = (None,) + parent_child(self.index)
            elif self.method == 'HCascade': func_HCascade(self)         
        find_Parent_child(self)
        
    def All_Nuclei(self):
        if self.method == 'HCascade': indexes = tuple([1,2,4,5,6,7,8,9,10,11,12,13,14]) + tuple([1.1,1.2,1.3,1.4])
        else:                         indexes = tuple([1,2,4,5,6,7,8,9,10,11,12,13,14])

        class All_Nuclei:
            Indexes = indexes[:]
            Names  = [self.dic_Name(index) for index in Indexes]

        return All_Nuclei()    
    
    def HCascade_Parents_Identifier(self, Nuclei_List):
  
        def fchild(ix): return Nuclei_Class(ix , self.method).child
        return [ix for ix in fchild(1) if fchild(ix) and bool(set(Nuclei_List) & set(fchild(ix))) ]  if self.method == 'HCascade' else [1]
        
    def remove_Thalamus_From_List(self , Nuclei_List):
        nuLs = Nuclei_List.copy()
        if 1 in nuLs: nuLs.remove(1)
        return nuLs
            
class Experiment_Folder_Search():    
    def __init__(self, General_Address='' , Experiment_Name = '' , subExperiment_Name='', mode='results'):
       
        class All_Experiments:
            def __init__(self, General_Address):
                        
                self.address = General_Address
                self.List = [s for s in os.listdir(General_Address) if 'exp' in s] if General_Address else []
            
        class Experiment:
            def __init__(self, Experiment_Name , AllExp_address):
                        
                self.name                = Experiment_Name
                self.address             = AllExp_address + '/' + Experiment_Name
                self.List_subExperiments = ''
                self.TagsList            = []

        def search_AllsubExp_inExp(Exp_address, mode):
                        
            class subExp():
                def __init__(self, name , address , TgC):

                    def find_Planes(address , name , TgC):

                        class SD:
                            def __init__(self, Flag = False , name='' , plane_name='', direction_name='' , TgC=0 , address=''):
                                self.Flag = Flag                        
                                if self.Flag: 
                                    self.tagList = np.append( ['Tag' + str(TgC) + '_' + plane_name], name.split('_') ) 
                                    self.tagIndex = str(TgC)
                                    if mode == 'results':
                                        self.subject_List = [a for a in os.listdir(address) if 'vimp' in a]
                                        self.subject_List.sort()
                                    self.name = plane_name
                                    self.direction = direction_name
                                    
                        multiPlanar = []
                        sdLst =  tuple(['sd0' , 'sd1' , 'sd2' , '2.5D_MV' , 'DT' , '2.5D_Sum' , '1.5D_Sum'])
                        PlNmLs = tuple(['Sagittal' , 'Coronal' , 'Axial', 'MV' , 'DT' , '2.5D_Sum' , '1.5D_Sum'])
                        if mode == 'results':
                            sdx = os.listdir(address + '/' + name)     
                            for sd, plane_name in zip(sdLst, PlNmLs): # ['sd0' , 'sd1' , 'sd2' , '2.5D_MV' , '2.5D_Sum' , '1.5D_Sum']:                                
                                multiPlanar.append( SD(True, name , sd , plane_name,TgC , address + '/' + name+'/' + sd) if sd in sdx else SD() )
                        else:
                            for sd, plane_name in zip(sdLst[:3], PlNmLs[:3]): # [:3]: # ['sd0' , 'sd1' , 'sd2' , '2.5D_MV' , '2.5D_Sum' , '1.5D_Sum']:                                
                                multiPlanar.append( SD(True, name , sd , plane_name,TgC , address + '/' + name+'/' + sd) if sd in sd else SD() )
                        return multiPlanar

                    self.name = name  
                    self.multiPlanar = find_Planes(address , name , TgC)
                    
            List_subExps = [a for a in os.listdir(Exp_address + '/' + mode) if ('subExp' in a) or ('sE' in a)] 
            List_subExps.sort()
            subExps = [subExp(name , Exp_address + '/' + mode , Ix)  for Ix, name in enumerate(List_subExps)]
            TagsList = [ np.append(['Tag' + str(Ix)],  name.split('_'))  for Ix, name in enumerate(List_subExps) ]

            return subExps, TagsList
                
        def func_Nuclei_Names():
            NumColumns = 19
            Nuclei_Names = np.append( ['subjects'] , list(np.zeros(NumColumns-1))  )
            Nuclei_Names[3] = ''
            def nuclei_Index_Integer(nIx):
                if nIx in range(15): return nIx
                elif nIx == 1.1:     return 15
                elif nIx == 1.2:     return 16
                elif nIx == 1.3:     return 17

            for nIx in Nuclei_Class().All_Nuclei().Indexes:
                Nuclei_Names[nuclei_Index_Integer(nIx)] = Nuclei_Class(index=nIx).name

            return Nuclei_Names        
        
        def search_1subExp(List_subExperiments):
            for a in List_subExperiments:
                if a.name == subExperiment_Name: 
                    return a

        self.All_Experiments = All_Experiments(General_Address)                 
        self.Experiment      = Experiment(Experiment_Name, self.All_Experiments.address)                                        
        self.Nuclei_Names    = func_Nuclei_Names()


        if Experiment_Name: 
            self.Experiment.List_subExperiments , self.Experiment.TagsList = search_AllsubExp_inExp(self.Experiment.address, mode)

        if subExperiment_Name:
            self.subExperiment = search_1subExp(self.Experiment.List_subExperiments)

def gpuSetting(GPU_Index):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_Index
    import tensorflow as tf
    from keras import backend as K
    K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))    
    # K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True )   ))
    return K

def listSubFolders(Dir, params):

    subFolders = [s for s in next(os.walk(Dir))[1] if 'ERROR' not in s]
    if params.WhichExperiment.Dataset.check_vimp_SubjectName:  subFolders = [s for s in subFolders if 'vimp' in s]

    subFolders.sort()
    return subFolders

def mkDir(Dir):
    if not os.path.isdir(Dir): os.makedirs(Dir)
    return Dir

def saveImage(Image , Affine , Header , outDirectory):
    mkDir(outDirectory.split(os.path.basename(outDirectory))[0])
    out = nib.Nifti1Image((Image).astype('float32'),Affine)
    out.get_header = Header
    nib.save(out , outDirectory)

def nibShow(*args):
    if len(args) > 1:
        for ax, im in enumerate(args):
            if ax == 0: a = nib.viewers.OrthoSlicer3D(im,title=str(ax))
            else: 
                b = nib.viewers.OrthoSlicer3D(im,title='2')
                a.link_to(b)
    else: 
        a = nib.viewers.OrthoSlicer3D(im)
    a.show()

def fixMaskMinMax(Image,name):
    if Image.max() > 1 or Image.min() < 0:
        print('Error in label values', 'min',Image.min() , 'max', Image.max() , '    ' , name.split('_PProcessed')[0])
        Image = np.float32(Image)
        Image = ( Image-Image.min() )/( Image.max() - Image.min() )
        
    return Image

def terminalEntries(UserInfo):

    for en in range(len(sys.argv)):
        entry = sys.argv[en]

        if entry.lower() in ('-g','--gpu'):  # gpu num
            UserInfo['simulation'].GPU_Index = sys.argv[en+1]

        elif entry.lower() in ('-sd','--slicingdim'):
            if sys.argv[en+1].lower() == 'all':
                UserInfo['simulation'].slicingDim = [2,1,0]

            elif sys.argv[en+1][0] == '[':
                B = sys.argv[en+1].split('[')[1].split(']')[0].split(",")
                UserInfo['simulation'].slicingDim = [int(k) for k in B]

            else:
                UserInfo['simulation'].slicingDim = [int(sys.argv[en+1])]

        elif entry in ('-Aug','--AugmentMode'):
            a = int(sys.argv[en+1])
            UserInfo['AugmentMode'] = True if a > 0 else False

        elif entry in ('-v','--verbose'):
            UserInfo['verbose'] = int(sys.argv[en+1])

        elif entry.lower() in ('-n','--nuclei'):  # nuclei index
            if sys.argv[en+1].lower() == 'all':
                _, UserInfo['simulation'].nucleus_Index,_ = NucleiSelection(ind = 1)

            elif sys.argv[en+1].lower() == 'allh':
                _, NucleiIndexes ,_ = NucleiSelection(ind = 1) 
                UserInfo['simulation'].nucleus_Index = tuple(NucleiIndexes) + tuple([1.1,1.2,1.3])

            elif sys.argv[en+1][0] == '[':
                B = sys.argv[en+1].split('[')[1].split(']')[0].split(",")
                UserInfo['simulation'].nucleus_Index = [int(k) for k in B]

            else:
                UserInfo['simulation'].nucleus_Index = [float(sys.argv[en+1])] # [int(sys.argv[en+1])]

        elif entry.lower() in ('-l','--loss'):
            UserInfo['lossFunctionIx'] = int(sys.argv[en+1])

        elif entry.lower() in ('-d','--dataset'):
            UserInfo['DatasetIx'] = int(sys.argv[en+1])

        elif entry.lower() in ('-e','--epochs'):
            UserInfo['simulation'].epochs = int(sys.argv[en+1])

        elif entry.lower() in ('-lr','--Learning_Rate'):
            UserInfo['simulation'].Learning_Rate = float(sys.argv[en+1])

        elif entry.lower() in ('-do','--DropoutValue'):
            UserInfo['DropoutValue'] = float(sys.argv[en+1])

        elif entry.lower() in ('-nl','--num_Layers'):
            UserInfo['simulation'].num_Layers = int(sys.argv[en+1])

        elif entry.lower() in ('-fm','--FirstLayer_FeatureMap_Num'):
            UserInfo['simulation'].FirstLayer_FeatureMap_Num = int(sys.argv[en+1])

        elif entry.lower() in ('-m','--Model_Method'):
            if int(sys.argv[en+1]) == 3:
                UserInfo['Model_Method'] = 'mUnet' 
            elif int(sys.argv[en+1]) == 1:
                UserInfo['Model_Method'] = 'Cascade' 
            elif int(sys.argv[en+1]) == 2: 
                UserInfo['Model_Method'] = 'HCascade' 

        elif entry.lower() in ('-ci','--CrossVal_Index'):
            UserInfo['CrossVal'].index = [sys.argv[en+1]]
                

                
            

    return UserInfo

def search_ExperimentDirectory(whichExperiment):

    def func_model_Tag(whichExperiment):
        model_Tag = ''
        if whichExperiment.HardParams.Model.Transfer_Learning.Mode:                               model_Tag += '_TF'
        if whichExperiment.Dataset.ReadTrain.ET   and not whichExperiment.Dataset.ReadTrain.Main: model_Tag += '_ET'
        if whichExperiment.Dataset.ReadTrain.CSFn and not whichExperiment.Dataset.ReadTrain.Main: model_Tag += '_CSFn'
        return model_Tag

    sdTag = '/sd' + str(whichExperiment.Dataset.slicingInfo.slicingDim)
    Exp_address = whichExperiment.Experiment.address
    SE          = whichExperiment.SubExperiment
    NucleusName = whichExperiment.Nucleus.name
    crossVal = whichExperiment.SubExperiment.crossVal

    def checkInputDirectory(Dir, NucleusName, sag_In_Cor,modeData):
        
        # sdTag2 = '/sd0' if sag_In_Cor else sdTag
        sdTag2 = '/sd'

        Read   = whichExperiment.Dataset.ReadTrain
        DirAug = Dir + '/Augments/' + Read.ReadAugments.Tag
        Dir_CV = whichExperiment.Experiment.address + '/crossVal'

        def Search_ImageFolder(Dir, NucleusName):

            def splitNii(s):
                return s.split('.nii.gz')[0]

            def Classes_Local(Dir):
                class deformation:
                    address = ''
                    testWarp = ''
                    testInverseWarp = ''
                    testAffine = ''

                class temp:
                    CropMask = ''
                    Cropped = ''
                    BiasCorrected = ''
                    Deformation = deformation
                    address = ''

                class tempLabel:
                    address = ''
                    Cropped = ''

                class label:
                    LabelProcessed = ''
                    LabelOriginal = ''
                    Temp = tempLabel
                    address = ''

                class newCropInfo:
                    OriginalBoundingBox = ''
                    PadSizeBackToOrig = ''

                class Files:
                    ImageOriginal = ''
                    ImageProcessed = ''
                    Label = label
                    Temp = temp
                    address = Dir
                    NewCropInfo = newCropInfo
                    subjectName = ''

                return Files

            def Look_Inside_Label_SF(Files, NucleusName):
                Files.Label.Temp.address = mkDir(Files.Label.address + '/temp')
                A = next(os.walk(Files.Label.address))
                for s in A[2]:
                    if NucleusName + '_PProcessed.nii.gz' in s: Files.Label.LabelProcessed = splitNii(s)
                    if NucleusName + '.nii.gz' in s: Files.Label.LabelOriginal = splitNii(s)

                if Files.Label.LabelOriginal and not Files.Label.LabelProcessed:
                    Files.Label.LabelProcessed = NucleusName + '_PProcessed'
                    _, _, FullNames = NucleiSelection(ind=1)
                    for name in FullNames: copyfile(Files.Label.address + '/' + name + '.nii.gz' , Files.Label.address + '/' + name + '_PProcessed.nii.gz')


                for s in A[1]:
                    if 'temp' in s:
                        Files.Label.Temp.address = Files.Label.address + '/' + s

                        for d in os.listdir(Files.Label.Temp.address):
                            if '_Cropped.nii.gz' in d: Files.Label.Temp.Cropped = splitNii(d)

                        # Files.Label.Temp.Cropped = [ d.split('.nii.gz')[0] for d in os.listdir(Files.Label.Temp.address) if '_Cropped.nii.gz' in d]
                    elif 'Label' in s: Files.Label.address = Dir + '/' + s

                return Files

            def Look_Inside_Temp_SF(Files):
                A = next(os.walk(Files.Temp.address))
                for s in A[2]:
                    if 'CropMask.nii.gz' in s: Files.Temp.CropMask = splitNii(s)
                    elif 'bias_corr.nii.gz' in s: Files.Temp.BiasCorrected = splitNii(s)
                    elif 'bias_corr_Cropped.nii.gz' in s: Files.Temp.Cropped = splitNii(s)
                    else: Files.Temp.origImage = splitNii(s)

                if 'deformation' in A[1]:
                    Files.Temp.Deformation.address = Files.Temp.address + '/deformation'
                    B = next(os.walk(Files.Temp.Deformation.address))
                    for s in B[2]:
                        if 'testWarp.nii.gz' in s: Files.Temp.Deformation.testWarp = splitNii(s)
                        elif 'testInverseWarp.nii.gz' in s: Files.Temp.Deformation.testInverseWarp = splitNii(s)
                        elif 'testAffine.txt' in s: Files.Temp.Deformation.testAffine = splitNii(s)

                if not Files.Temp.Deformation.address: Files.Temp.Deformation.address = mkDir(Files.Temp.address + '/deformation')

                return Files

            def check_IfImageFolder(Files):
                A = next(os.walk(Files.address))
                for s in A[2]:
                    if 'PProcessed.nii.gz' in s: Files.ImageProcessed = splitNii(s)
                    if '.nii.gz' in s and 'PProcessed.nii.gz' not in s: Files.ImageOriginal = splitNii(s)

                if Files.ImageOriginal or Files.ImageProcessed:
                    for s in A[1]:
                        if 'temp' in s: Files.Temp.address = mkDir(Dir + '/' + s)
                        elif 'Label' in s: Files.Label.address = Dir + '/' + s

                    if Files.ImageOriginal and not Files.ImageProcessed:
                        Files.ImageProcessed = 'PProcessed'
                        copyfile(Dir + '/' + Files.ImageOriginal + '.nii.gz' , Dir + '/' + Files.ImageProcessed + '.nii.gz')

                if not Files.Temp.address: Files.Temp.address = mkDir(Dir + '/temp')

                return Files

            Files = Classes_Local(Dir)
            Files = check_IfImageFolder(Files)

            if Files.ImageOriginal or Files.ImageProcessed:
                if os.path.exists(Files.Label.address): Files = Look_Inside_Label_SF(Files, NucleusName)
                if os.path.exists(Files.Temp.address):  Files = Look_Inside_Temp_SF(Files)

            return Files                
        class Input:
            address = os.path.abspath(Dir)
            Subjects = {}

        # Input = Input_cls()

        def LoopReadingData(Input, Dirr):
            if os.path.exists(Dirr):
                SubjectsList = next(os.walk(Dirr))[1]

                if whichExperiment.Dataset.check_vimp_SubjectName: SubjectsList = [s for s in SubjectsList if 'vimp' in s]

                for s in SubjectsList:
                    Input.Subjects[s] = Search_ImageFolder(Dirr + '/' + s , NucleusName)
                    Input.Subjects[s].subjectName = s

            return Input

        def load_CrossVal_Data(Input , Dir):            
            if os.path.exists(Dir):
                CV_list = crossVal.index if (modeData == 'test') else [s for s in os.listdir(Dir) if not (s in crossVal.index) ]

                for x in CV_list: 
                    Input = LoopReadingData(Input, Dir + x)

                    if Read.ReadAugments.Mode and not (modeData == 'test'): 
                        Input = LoopReadingData(Input , Dir + x + '/Augments' + sdTag2)

            return Input

        Input = LoopReadingData(Input, Dir)

        # SRI_flag_test = False if (Read.Main or Read.ET) and (modeData == 'test') else True
        # SRI_flag_test = True
        
        if Read.Main : Input = LoopReadingData(Input, Dir + '/Main')
        if Read.ET   : Input = LoopReadingData(Input, Dir + '/ET')                   
        if Read.SRI  : Input = LoopReadingData(Input, Dir + '/SRI')
        if Read.CSFn : Input = LoopReadingData(Input, Dir + '/CSFn')
        # if Read.SRI and SRI_flag_test: Input = LoopReadingData(Input, Dir + '/SRI')
        
        if crossVal.Mode:
            if Read.Main:  Input = load_CrossVal_Data(Input , Dir_CV + '/Main/')
            if Read.ET:    Input = load_CrossVal_Data(Input , Dir_CV + '/ET/')
            if Read.SRI:   Input = load_CrossVal_Data(Input , Dir_CV + '/SRI/')
            if Read.CSFn : Input = load_CrossVal_Data(Input , Dir_CV + '/CSFn/')    



        if Read.ReadAugments.Mode and not (modeData == 'test'):
            
            if Read.Main and os.path.exists(DirAug + '/Main' + sdTag2): Input = LoopReadingData(Input, DirAug + '/Main' + sdTag2)
            if Read.ET   and os.path.exists(DirAug + '/ET'   + sdTag2): Input = LoopReadingData(Input, DirAug + '/ET'   + sdTag2)
            if Read.SRI  and os.path.exists(DirAug + '/SRI'  + sdTag2): Input = LoopReadingData(Input, DirAug + '/SRI'  + sdTag2)
            if Read.CSFn and os.path.exists(DirAug + '/CSFn' + sdTag2): Input = LoopReadingData(Input, DirAug + '/CSFn' + sdTag2)
                             
             
            if Read.Main and os.path.exists(Dir + '/Main/Augments' + sdTag2): Input = LoopReadingData(Input, Dir + '/Main/Augments' + sdTag2)
            if Read.ET   and os.path.exists(Dir + '/ET/Augments'   + sdTag2): Input = LoopReadingData(Input, Dir + '/ET/Augments'   + sdTag2)
            if Read.SRI  and os.path.exists(Dir + '/SRI/Augments'  + sdTag2): Input = LoopReadingData(Input, Dir + '/SRI/Augments'  + sdTag2)
            if Read.CSFn and os.path.exists(Dir + '/CSFn/Augments' + sdTag2): Input = LoopReadingData(Input, Dir + '/CSFn/Augments' + sdTag2)

        return Input

    def add_Sagittal_Cases(whichExperiment , train , test , NucleusName):
        if whichExperiment.Nucleus.Index[0] == 1 and whichExperiment.Dataset.slicingInfo.slicingDim == 2:
            train.Input_Sagittal = checkInputDirectory(train.address, NucleusName, True, 'train') 
            test.Input_Sagittal  = checkInputDirectory(test.address , NucleusName, True, 'test') 
        return train , test
    
    class train:
        address        = Exp_address + '/train'
        Model          = Exp_address + '/models/' + SE.name                   + '/' + NucleusName  + sdTag
        Model_Thalamus = Exp_address + '/models/' + SE.name                   + '/' + '1-THALAMUS' + sdTag
        Model_3T       = Exp_address + '/models/' + SE.name_Init_from_3T      + '/' + NucleusName  + sdTag
        Model_7T       = Exp_address + '/models/' + SE.name_Init_from_7T      + '/' + NucleusName  + sdTag
        Model_InitTF   = Exp_address + '/models/' + SE.name.split('_TF_')[0]  + '/' + NucleusName  + sdTag
        model_Tag = func_model_Tag(whichExperiment)
        Input     = checkInputDirectory(address, NucleusName,False,'train')
    
    class test:
        address = Exp_address + '/test'
        Result  = Exp_address + '/results/' + SE.name + sdTag
        Input   = checkInputDirectory(address, NucleusName,False,'test')


    train , test = add_Sagittal_Cases(whichExperiment , train , test , NucleusName)

    class Directories:
        Train = train()
        Test  = test()

    return Directories()

def imShow(*args):
    _, axes = plt.subplots(1,len(args))
    for ax, im in enumerate(args):
        axes[ax].imshow(im,cmap='gray')

    # a = nib.viewers.OrthoSlicer3D(im,title='image')

    plt.show()

    return True

def mDice(msk1,msk2):
    intersection = msk1*msk2
    return intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)

"""
def Loading_UserInfo(DirLoad, method):

    def loadReport(DirSave, name, method):

        def loadPickle(Dir):
            f = open(Dir,"wb")
            data = pickle.load(f)
            f.close()
            return data

        if 'pickle' in method:
            return loadPickle(DirSave + '/' + name + '.pkl')
        # elif 'mat' in method:
            # return mat4py.loadmat(DirSave + '/' + name + '.pkl')

    def dict2obj(d):
        if isinstance(d, list):
            d = [dict2obj(x) for x in d]
        if not isinstance(d, dict):
            return d
        class C(object):
            pass
        o = C()
        for k in d:
            o.__dict__[k] = dict2obj(d[k])
        return o

    UserInfo = loadReport(DirLoad, 'UserInfo', method)
    UserInfo = dict2obj( UserInfo )

    a = UserInfo['InputDimensions'].replace(',' ,'').split('[')[1].split(']')[0].split(' ')
    UserInfo['InputDimensions'] = [int(ai) for ai in a]
    return UserInfo
"""

def findBoundingBox(PreStageMask):
    objects = measure.regionprops(measure.label(PreStageMask))

    L = len(PreStageMask.shape)
    if len(objects) > 1:
        area = []
        for obj in objects: area = np.append(area, obj.area)

        Ix = np.argsort(area)
        bbox = objects[ Ix[-1] ].bbox

    else:
        bbox = objects[0].bbox

    BB = [ [bbox[d] , bbox[L + d] ] for d in range(L)]

    return BB

def Saving_UserInfo(DirSave, params):
                            
    # DirSave = '/array/ssd/msmajdi/experiments/keras/exp7_cascadeV1/models/sE11_Cascade_wRot7_6cnts_sd2/4-VA'
    User_Info = {            
        'num_Layers'     : int(params.WhichExperiment.HardParams.Model.num_Layers),
        'Trained_SRI'    : params.UserInfo['ReadTrain'].SRI,
        'Trained_ET'     : params.UserInfo['ReadTrain'].ET,
        'Trained_Main'   : params.UserInfo['ReadTrain'].Main,
        'Model_Method'   : params.UserInfo['Model_Method'],
        'FromThalamus'   : params.UserInfo['InitializeB'].FromThalamus,
        'FromOlderModel' : params.UserInfo['InitializeB'].FromOlderModel,
        'From_3T'        : params.UserInfo['InitializeB'].From_3T,
        'Learning_Rate'  : params.UserInfo['simulation'].Learning_Rate,
        'Normalizae'     : params.UserInfo['simulation'].NormalizaeMethod,
        'slicing_Dim'    : params.UserInfo['simulation'].slicingDim[0],
        'batch'          : int(params.UserInfo['simulation'].batch_size),
        'FeatureMaps'    : int(params.UserInfo['simulation'].FirstLayer_FeatureMap_Num),
        'Mult_Thalmaus'  : params.UserInfo['simulation'].Multiply_By_Thalmaus,
        'Weighted_Class' : params.UserInfo['simulation'].Weighted_Class_Mode,
        'Dropout'        : params.UserInfo['DropoutValue'],
        'gapDilation'    : int(params.UserInfo['gapDilation']),
        'ImClosePrediction' : params.UserInfo['simulation'].ImClosePrediction,
        'InputPadding_Mode' : params.UserInfo['InputPadding'].Automatic,
        'InputPadding_Dims' : [int(s) for s in params.WhichExperiment.HardParams.Model.InputDimensions],
    }
    mkDir(DirSave)
    with open(DirSave + '/UserInfo.json', "w") as j:
        j.write(json.dumps(User_Info))

    # with open(DirSave + '/UserInfo.json', "r") as j:
    #     data = json.load(j)

def closeMask(mask,cnt):
    struc = ndimage.generate_binary_structure(3,2)
    if cnt > 1: struc = ndimage.iterate_structure(struc, cnt)
    return ndimage.binary_closing(mask, structure=struc)  
