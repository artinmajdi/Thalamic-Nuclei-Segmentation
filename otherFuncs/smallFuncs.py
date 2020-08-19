import nibabel as nib
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
from skimage import measure
from copy import deepcopy
import json
from scipy import ndimage
from tqdm import tqdm
from modelFuncs import Metrics as metrics
from glob import glob

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

class Nuclei_Class():

    def __init__(self, index=1, method = 'Cascade'):

        def nucleus_name_func(index):
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
                14: '14-MTT'}
            return switcher.get(index, 'wrong index')
        self.name = nucleus_name_func(index)
        self.nucleus_name_func = nucleus_name_func
        self.method = method
        self.index = index
        self.parent , self.child = ( None, [2,4,5,6,7,8,9,10,11,12,13,14] ) if self.index == 1 else (1,None)

    def All_Nuclei(self):                      
        class All_Nuclei:
            Indexes = tuple([1,2,4,5,6,7,8,9,10,11,12,13,14])
            Names  = [self.nucleus_name_func(index) for index in Indexes]

        return All_Nuclei()

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

            # TODO this is temporary, to only save the subexperiments I need
            # List_subExps = [a for a in List_subExps if 'Res_Unet2_NL3_LS_MyDice_US1_wLRScheduler_Main_Ps_ET_7T_Init_Rn_test_ET_3T' in a]

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
                elif nIx == 1.4:     return 18

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
    # tf.compat.v1.disable_v2_behavior()

    # K.set_session(tf.compat.v1.Session(   config=tf.compat.v1.ConfigProto( allow_soft_placement=True , gpu_options=tf.compat.v1.GPUOptions(allow_growth=True) )   ))
    K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True )   ))
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
    """ Inputs:  Image , Affine , Header , outDirectory """
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

        elif entry.lower() in ('-pi','--permutation_Index'):
            UserInfo['permutation_Index'] = int(sys.argv[en+1])

        elif entry.lower() in ('-fm','--FirstLayer_FeatureMap_Num'):
            UserInfo['simulation'].FirstLayer_FeatureMap_Num = int(sys.argv[en+1])

        elif entry.lower() in ('-m','--Model_Method'):
            if int(sys.argv[en+1]) == 1:
                UserInfo['Model_Method'] = 'Cascade'
            elif int(sys.argv[en+1]) == 2:
                UserInfo['Model_Method'] = 'HCascade'
            elif int(sys.argv[en+1]) == 3:
                UserInfo['Model_Method'] = 'mUnet'
            elif int(sys.argv[en+1]) == 4:
                UserInfo['Model_Method'] = 'normal'
                UserInfo['architectureType'] = 'FCN'

            # elif int(sys.argv[en+1]) == 5:
            #     UserInfo['Model_Method'] = 'FCN_with_SkipConnection'
                # UserInfo['architectureType'] = 'FCN_with_SkipConnection'

        elif entry.lower() in ('-cv','--CrossVal_Index'):
            UserInfo['CrossVal'].index = [sys.argv[en+1]]
            print('CrossVal' , UserInfo['CrossVal'].index)
    
    return UserInfo

def search_ExperimentDirectory(whichExperiment):

    sdTag = '/sd' + str(whichExperiment.Dataset.slicingInfo.slicingDim)
    Exp_address = whichExperiment.Experiment.exp_address
    SEname      = whichExperiment.Experiment.subexperiment_name
    NucleusName = whichExperiment.Nucleus.name
    

    def checkInputDirectory(Dir, NucleusName, sag_In_Cor,modeData):

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

            def search_inside_subject_folder_label(Files, NucleusName):
                
                A = next(os.walk(Files.Label.address))

                # Single Class 
                if 'MultiClass' not in NucleusName:
                    for s in A[2]:
                        if NucleusName + '_PProcessed.nii.gz' in s: 
                            Files.Label.LabelProcessed = NucleusName + '_PProcessed.nii.gz'
                        elif NucleusName + '.nii.gz' in s: 
                            Files.Label.LabelOriginal = NucleusName
                
                    if not Files.Label.LabelProcessed:
                        Files.Label.LabelProcessed = NucleusName + '_PProcessed.nii.gz'
                        copyfile(Files.Label.address + '/' + NucleusName + '.nii.gz' , Files.Label.address + '/' + NucleusName + '_PProcessed.nii.gz')

                    cpd_lst = glob(Files.Label.Temp.address + '/' + NucleusName + '_Cropped.nii.gz')
                    if cpd_lst: 
                        Files.Label.Temp.Cropped = NucleusName + '_Cropped'

                # Multi Class
                else:
                    for nucleus in whichExperiment.Nucleus.FullNames:
                        if nucleus + '.nii.gz' in A[2]:
                            copyfile(Files.Label.address + '/' + nucleus + '.nii.gz' , Files.Label.address + '/' + nucleus + '_PProcessed.nii.gz')

                return Files

            def search_inside_subject_folder_temp(Files):
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

                if not Files.Temp.Deformation.address: 
                    Files.Temp.Deformation.address = mkDir(Files.Temp.address + '/deformation')

                return Files

            def search_inside_subject_folder(Files):
                for s in [subj for subj in os.listdir(Files.address) if '.nii.gz' in subj]:
                    if 'PProcessed.nii.gz' in s:
                        Files.ImageProcessed = 'PProcessed'
                    else:
                        Files.ImageOriginal  = s.split('.nii.gz')[0]
                        Files.Temp.address   = mkDir(Files.address + '/temp')
                        Files.Label.address  = Files.address + '/Label'
                        Files.Label.Temp.address = mkDir(Files.address  + '/Label/temp')

                if (not Files.ImageProcessed) and Files.ImageOriginal:
                    Files.ImageProcessed = 'PProcessed'
                    copyfile(Dir + '/' + Files.ImageOriginal + '.nii.gz' , Dir + '/PProcessed.nii.gz')

                return Files

            Files = Classes_Local(Dir)
            Files = search_inside_subject_folder(Files)

            if Files.ImageOriginal:
                if os.path.exists(Files.Label.address): Files = search_inside_subject_folder_label(Files, NucleusName)
                if os.path.exists(Files.Temp.address):  Files = search_inside_subject_folder_temp(Files)

            return Files
        class Input:
            address = os.path.abspath(Dir)
            Subjects = {}

        def LoopReadingData(Input, Dirr):
            if os.path.exists(Dirr):
                SubjectsList = next(os.walk(Dirr))[1]

                if whichExperiment.Dataset.check_vimp_SubjectName:  
                    SubjectsList = [s for s in SubjectsList if ('vimp' in s)]

                for s in SubjectsList:
                    Input.Subjects[s] = Search_ImageFolder(Dirr + '/' + s , NucleusName)
                    Input.Subjects[s].subjectName = s

            return Input

        if not(modeData == 'train' and whichExperiment.TestOnly):
            Dir = whichExperiment.Experiment.train_address if modeData == 'train' else whichExperiment.Experiment.test_address
            Input = LoopReadingData(Input, Dir)

            if whichExperiment.Experiment.ReadAugments_Mode and not (modeData == 'test'):
                Input = LoopReadingData(Input, Dir + '/Augments/')

        return Input

    def add_Sagittal_Cases(whichExperiment , train , test , NucleusName):
        if whichExperiment.Nucleus.Index[0] == 1 and whichExperiment.Dataset.slicingInfo.slicingDim == 2:
            train.Input_Sagittal = checkInputDirectory(train.address, NucleusName, True, 'train')
            test.Input_Sagittal  = checkInputDirectory(test.address , NucleusName, True, 'test')
        return train , test

    FM = '/FM' + str(whichExperiment.HardParams.Model.Layer_Params.FirstLayer_FeatureMap_Num)
    class train:
        address   = Exp_address + '/train'
        Model     = Exp_address + '/models/' + SEname                   + '/' + NucleusName  + FM + sdTag
        model_Tag = ''
        Input     = checkInputDirectory(address, NucleusName,False,'train')

    class test:
        address = Exp_address + '/test'
        Result  = Exp_address + '/results/' + SEname + sdTag
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
        'Model_Method'   : params.UserInfo['Model_Method'],
        'Learning_Rate'  : params.UserInfo['simulation'].Learning_Rate,
        'slicing_Dim'    : params.UserInfo['simulation'].slicingDim[0],
        'batch'          : int(params.UserInfo['simulation'].batch_size),
        'InputPadding_Mode' : params.UserInfo['InputPadding']().Automatic,
        'InputPadding_Dims' : [int(s) for s in params.WhichExperiment.HardParams.Model.InputDimensions],
    }
    mkDir(DirSave)
    with open(DirSave + '/UserInfo.json', "w") as j:
        j.write(json.dumps(User_Info))

def closeMask(mask,cnt):
    struc = ndimage.generate_binary_structure(3,2)
    if cnt > 1: struc = ndimage.iterate_structure(struc, cnt)
    return ndimage.binary_closing(mask, structure=struc)

def dir_check(directory):
    if directory[-1] != '/':
        directory = directory + '/'
    return directory

def apply_MajorityVoting(params):

    def func_manual_label(subject,nucleusNm):
        class manualCs:
            def __init__(self, flag, label):
                self.Flag = flag
                self.Label = label

        dirr = subject.address + '/Label/' + nucleusNm + '_PProcessed.nii.gz'

        if os.path.isfile(dirr):
            label = fixMaskMinMax(nib.load(dirr).get_data(),nucleusNm)
            manual = manualCs(flag=True, label=label)

        else: manual = manualCs(flag=False,label='')

        return manual

    # address = params.WhichExperiment.Experiment.exp_address + '/results/' + params.WhichExperiment.Experiment.subexperiment_name + '/'

    a = Nuclei_Class().All_Nuclei()
    num_classes = params.WhichExperiment.HardParams.Model.MultiClass.num_classes

    for sj in tqdm(params.directories.Test.Input.Subjects):
        subject = params.directories.Test.Input.Subjects[sj]
        address = subject.address + '/' + params.UserInfo['thalamic_side'].active_side + '/'

        VSI, Dice, HD= np.zeros((num_classes,2)) , np.zeros((num_classes,2)) , np.zeros((num_classes,2))
        for cnt, (nucleusNm , nucleiIx) in enumerate(zip(a.Names , a.Indexes)):

            ix , pred3Dims = 0 , ''
            im = nib.load(subject.address + '/' + subject.ImageProcessed + '.nii.gz')

            manual = func_manual_label(subject, nucleusNm)
            for sdInfo in ['sd0', 'sd1' , 'sd2']:
                address_nucleus = address + sdInfo + '/' + nucleusNm + '.nii.gz'
                if os.path.isfile(address_nucleus):

                    pred = nib.load(address_nucleus).get_data()[...,np.newaxis]
                    pred3Dims = pred if ix == 0 else np.concatenate((pred3Dims,pred),axis=3)
                    ix += 1

            if ix > 0:
                predMV = pred3Dims.sum(axis=3) >= 2
                saveImage( predMV , im.affine, im.header, address + '2.5D_MV/' + nucleusNm + '.nii.gz')

                if manual.Flag:
                    VSI[cnt,:]  = [nucleiIx , metrics.VSI_AllClasses(predMV, manual.Label).VSI()]
                    HD[cnt,:]   = [nucleiIx , metrics.HD_AllClasses(predMV, manual.Label).HD()]
                    Dice[cnt,:] = [nucleiIx , mDice(predMV, manual.Label)]

        if Dice[:,1].sum() > 0:
            np.savetxt( address + '2.5D_MV/VSI_All.txt'  ,VSI  , fmt='%1.1f %1.4f')
            np.savetxt( address + '2.5D_MV/HD_All.txt'   ,HD   , fmt='%1.1f %1.4f')
            np.savetxt( address + '2.5D_MV/Dice_All.txt' ,Dice , fmt='%1.1f %1.4f')

def extracting_the_biggest_object(pred_Binary):

    objects = measure.regionprops(measure.label(pred_Binary))

    # L = len(pred_Binary.shape)
    if len(objects) > 1:
        area = []
        for obj in objects: area = np.append(area, obj.area)

        Ix = np.argsort(area)
        obj = objects[ Ix[-1] ]

        fitlered_pred = np.zeros(pred_Binary.shape)
        for cds in obj.coords:
            fitlered_pred[tuple(cds)] = True

        return fitlered_pred

    else:
        return pred_Binary

def test_precision_recall():
    import pandas as pd


    dir = '/array/ssd/msmajdi/experiments/keras/exp6/results/sE12_Cascade_FM20_Res_Unet2_NL3_LS_MyDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_a/sd2/vimp2_967_08132013_KW/'
    dirM = '/array/ssd/msmajdi/experiments/keras/exp6/crossVal/Main/a/vimp2_967_08132013_KW/Label/'

    Names = Nuclei_Class(index=1,method='Cascade').All_Nuclei().Names

    write_flag = False
    PR = {}
    if write_flag: df = pd.DataFrame()
    if write_flag: writer = pd.ExcelWriter(path=dir + 'Precision_Recall.xlsx', engine='xlsxwriter')


    for ind in range(13):

        nucleus_name = Names[ind].split('-')[1]
        msk = nib.load(dir  + Names[ind] + '.nii.gz').get_data()
        mskM = nib.load(dirM  + Names[ind] + '_PProcessed.nii.gz').get_data()

        # plt.plot(np.unique(msk))

        precision, recall = metrics.Precision_Recall_Curve(y_true=mskM,y_pred=msk, Show=True, name=nucleus_name, directory=dir)

        if write_flag:
            df = pd.DataFrame.from_dict({'precision':precision , 'recall':recall})
            df.to_excel(writer, sheet_name=nucleus_name)

    if write_flag: writer.save()

def test_extract_biggest_object():

    import nibabel as nib
    import smallFuncs
    from skimage import measure
    import numpy as np
    import matplotlib.pyplot as plt
    import os, sys
    from tqdm import tqdm


    dir_predictions = '/array/ssd/msmajdi/experiments/keras/exp6_uncropped/results/sE12_Cascade_FM20_Res_Unet2_NL3_LS_MyDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CVs_all/sd2/'
    main_directory = '/array/ssd/msmajdi/experiments/keras/exp6_uncropped/crossVal/Main/' # c/vimp2_988_08302013_CB/PProcessed.nii.gz'

    for cv in tqdm(['a/', 'b/' , 'c/' , 'd/' , 'e/', 'f/']):

        if not os.path.exists(main_directory + cv): continue

        subjects = [s for s in os.listdir(main_directory + cv) if 'vimp' in s]
        for subj in tqdm(subjects):

            im = nib.load(main_directory + cv + subj + '/PProcessed.nii.gz').get_data()
            label = nib.load(main_directory + cv + subj + '/Label/1-THALAMUS_PProcessed.nii.gz').get_data()
            label = smallFuncs.fixMaskMinMax(label,subj)
            OP = nib.load(dir_predictions + subj + '/1-THALAMUS.nii.gz')
            original_prediction = OP.get_data()


            objects = measure.regionprops(measure.label(original_prediction))

            L = len(original_prediction.shape)
            if len(objects) > 1:
                area = []
                for obj in objects: area = np.append(area, obj.area)

                Ix = np.argsort(area)
                obj = objects[ Ix[-1] ]

                fitlered_prediction = np.zeros(original_prediction.shape)
                for cds in obj.coords:
                    fitlered_prediction[tuple(cds)] = True

                Dice = np.zeros(2)
                Dice[0], Dice[1] = 1, smallFuncs.mDice(fitlered_prediction > 0.5 , label > 0.5)
                np.savetxt(dir_predictions + subj + '/Dice_1-THALAMUS_biggest_obj.txt' ,Dice,fmt='%1.4f')

                smallFuncs.saveImage(fitlered_prediction , OP.affine , OP.header , dir_predictions + subj + '/1-THALAMUS_biggest_obj.nii.gz')

        else:
            image = objects[0].image

class SNR_experiment():

    def __init__(self):
        pass

    def script_adding_WGN(self, directory='/mnt/sda5/RESEARCH/PhD/Thalmaus_Dataset/SNR_Tests/vimp2_ANON695_03132013/', SD_list=np.arange(1,80,5), run_network=True):

        def add_WGN(input=[], noise_mean=0,noise_std=1):
            from numpy.fft import fftshift, fft2
            gaussian_noise_real = np.random.normal(loc=noise_mean , scale = noise_std, size=input.shape)
            gaussian_noise_imag = np.random.normal(loc=noise_mean , scale = noise_std, size=input.shape)
            gaussian_noise = gaussian_noise_real + 1j*gaussian_noise_imag

            # template = nib.load('general/RigidRegistration/cropped_origtemplate.nii.gz').get_data()
            # psd_template = np.mean(abs(fftshift(fft2(template/template.max())))**2)

            # psd_template, max_template = 325, 12.66
            psd_signal = np.mean(abs(fftshift(fft2(input)))**2)

            # psd_noise_image = psd_signal - psd_template * (input.max()**2)
            psd_noise = np.mean(abs(fftshift(fft2(gaussian_noise)))**2)
            # psd_noise       = psd_noise_added + psd_noise_image

            SNR = 10*np.log10(psd_signal/psd_noise)  # SNR = PSD[s]/PSD[n]  if x = s + n
            # SNR = 10*np.log10(np.mean(input**2)/np.mean(abs(gaussian_noise)**2))  # SNR = E[s^2]/E[n^2]  if x = s + n
            return abs(input + gaussian_noise) , SNR

        directory2 = os.path.dirname(directory).replace(' ','\ ')
        os.system("mkdir {0}/vimp2_orig_SNR_10000 ; mv {0}/* {0}/vimp2_orig_SNR_10000/ ".format(directory2))
        subject_name = 'vimp2_orig'
        dir_original_image = directory + subject_name + '_SNR_10000'

        SNR_List = []
        np.random.seed(0)
        for noise_std in SD_list:

            imF = nib.load(dir_original_image + '/PProcessed.nii.gz')
            im = imF.get_data()

            noisy_image, SNR = add_WGN(input=im, noise_mean=0,noise_std=noise_std)
            SNR = int(round(SNR))

            if SNR not in SNR_List:
                SNR_List.append(SNR)
                print('SNR:',int(round(SNR)), 'std:',noise_std)

                dir_noisy_image = directory + 'vimp2_noisy_SNR_' + str(SNR)
                saveImage(noisy_image , imF.affine , imF.header , dir_noisy_image + '/PProcessed.nii.gz')

                os.system('cp -r %s %s'%(dir_original_image.replace(' ','\ ') + '/Label', dir_noisy_image.replace(' ','\ ') + '/Label'))

                if run_network:
                    os.system( "python main.py --test %s"%(dir_noisy_image.replace(' ','\ ') + '/PProcessed.nii.gz') )

    def read_all_Dices_and_SNR(self, directory=''):
        Dices = {}
        for subj in [s for s in os.listdir(directory) if os.path.isdir(directory + s)]:

            SNR = int(subj.split('_SNR_')[-1])

            Dices[SNR] = pd.read_csv( directory + subj + '/left/2.5D_MV/Dice_All.txt',index_col=0,header=None,delimiter=' ',names=[SNR]).values.reshape([-1]) if os.path.isfile(directory + subj + '/left/2.5D_MV/Dice_All.txt') else np.zeros(13)

        df = pd.DataFrame(Dices,index=Thalamus_Sub_Functions().All_Nuclei().Names)
        df = df.transpose()
        df.columns.name = 'nucleus'
        df.index.name = 'SNR'
        df = df.sort_values(['SNR'], ascending=[False])
        df.to_csv(directory + 'Dice_vs_SNR.csv')

        return df

    def loop_all_subjects_read_Dice_SNR(self, directory='/mnt/sda5/RESEARCH/PhD/Thalmaus_Dataset/SNR_Tests/'):

        writer = pd.ExcelWriter(directory + 'Dice_vs_SNR.xlsx',engine='xlsxwriter')
        for subject in [s for s in os.listdir(directory) if os.path.isdir(directory + s) and 'vimp2_' in s]:
            print(subject)
            df = self.read_all_Dices_and_SNR(directory=directory + subject + '/')
            df.to_excel(writer, sheet_name=subject)

        writer.save()

class Thalamus_Sub_Functions():
    def __init__(self):
        pass

    def measure_metrics(self, Dir_manual='', Dir_prediction='', metrics=['DICE'], save=False):

        if not (os.path.isdir(Dir_prediction) and os.path.isdir(Dir_manual)): raise Warning('directory does not exist'.upper())
        Measurements = {s:[] for s in metrics}
        for nuclei in Nuclei_Class().All_Nuclei().Names:
            pred = nib.load(Dir_prediction + nuclei + '.nii.gz').get_data()
            manual = nib.load(Dir_manual + nuclei + '_PProcessed.nii.gz').get_data()

            if 'DICE' in metrics: Measurements['DICE'].append([nuclei, mDice(pred, manual)])

        if save:
            for mt in metrics: np.savetxt(Dir_prediction + 'All_' + mt + '.txt', Measurements[mt] , fmt='%1.1f %1.4f')

        return Measurements

    def nucleus_name(self, index=1):
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
            14: '14-MTT'}
        return switcher.get(index, 'wrong index')

    def All_Nuclei(self):
        indexes = tuple([1,2,4,5,6,7,8,9,10,11,12,13,14])

        class All_Nuclei:
            Indexes = indexes[:]
            Names  = [self.nucleus_name(index) for index in Indexes]

        return All_Nuclei()

    def run_network(self,directory='mnt/PProcessed.nii.gz', thalamic_side='--left', modality='--wmn', gpu="None"):
        os.system( 'python main.py --test %s %s %s --gpu %s'%(directory, thalamic_side, modality, gpu) )
