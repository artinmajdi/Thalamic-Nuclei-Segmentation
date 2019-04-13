import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import otherFuncs.smallFuncs as smallFuncs
from scipy import ndimage
from skimage.feature import canny

def sliceDim(SD):
    if   SD == 0: return ([1,2,0] , [2,0,1])
    elif SD == 1: return ([2,0,1] , [1,2,0])
    elif SD == 2: return ([0,1,2] , [0,1,2])
    

def applyMain(Dir,mode):
    
    def RunAllFunctions(Directory):
        im = nib.load(Directory + '1-THALAMUS' + mode + '.nii.gz')

        class inputInfo:
            def __init__(self, fullIndexes, FullNames ):
                self.fullIndexes = fullIndexes
                self.FullNames = FullNames


        _, HierarchicalIndexes, HierarchicalNames = smallFuncs.NucleiSelection(ind=1.9)
        HierarchicalNames = [nm.split('_ImClosed')[0] for nm in HierarchicalNames]
        
        Names = dict()
        for ix in HierarchicalIndexes:
            name, fullIndexes, FullNames = smallFuncs.NucleiSelection(ind=ix)
            if '_ImClosed' in name: name = name.split('_ImClosed')[0]
            Names[name] = inputInfo(fullIndexes=fullIndexes, FullNames=FullNames)

        # Names = dict({1:['posterior',[8,9,10]], 2:['lateral',[4,5,6,7]] , 3:['Anterior',[2]] , 4:['Medial',[11,12,13]] })

        def closeMask(mask):
            struc = ndimage.generate_binary_structure(3,2)
            return ndimage.binary_closing(mask, structure=struc)

        def saving4SuperNuclei():
            print('    saving 4 Super Nuclei')
            for superNuclei in HierarchicalNames:
                for cnt, subNuclei in enumerate(Names[superNuclei].FullNames):
                    msk = nib.load(Directory + subNuclei + mode + '.nii.gz').get_data()
                    Mask = msk if cnt == 0 else Mask + msk

                smallFuncs.saveImage( Mask > 0 , im.affine , im.header, Directory + 'Hierarchical/' + superNuclei + mode + '.nii.gz')
                smallFuncs.saveImage( closeMask(Mask > 0) , im.affine , im.header, Directory + superNuclei + '_ImClosed' + mode + '.nii.gz')

        def saving4SuperNuclei_WithDifferentLabels():
            print('    saving 4 Super Nuclei')
            for superNuclei in HierarchicalNames:
                for cnt, subNuclei in enumerate(Names[superNuclei].FullNames):
                    msk = nib.load(Directory + subNuclei + mode + '.nii.gz').get_data()
                    Mask = msk if cnt == 0 else Mask + (cnt+1)*msk

                smallFuncs.saveImage( Mask , im.affine , im.header, Directory + superNuclei + mode + '_DifferentLabels.nii.gz')
                # smallFuncs.saveImage( closeMask(Mask) , im.affine , im.header, Directory + superNuclei + '_ImClosed' + mode + '_DifferentLabels.nii.gz')


        def creatingFullMaskWithAll4Supernuclei():
            print('    creating Full Mask With All 4 Super Nuclei')
            for cnt, superNuclei in enumerate(HierarchicalNames):
                msk = nib.load(Directory + 'Hierarchical/' + superNuclei + mode + '.nii.gz').get_data()
                Mask = msk if cnt == 0 else Mask + (cnt+1)*msk

                msk = nib.load(Directory + superNuclei + '_ImClosed' + mode + '.nii.gz').get_data()
                MaskClosed = msk if cnt == 0 else MaskClosed + (cnt+1)*msk

            smallFuncs.saveImage(Mask , im.affine , im.header, Directory + 'Hierarchical/All_4MainNuclei' + mode + '.nii.gz')
            smallFuncs.saveImage(MaskClosed , im.affine , im.header, Directory + 'Hierarchical/All_4MainNuclei_ImClosed' + mode + '.nii.gz')

        def ImClosingAllNuclei():
            print('    ImClosing All Nuclei')
            _, _, AllNames = smallFuncs.NucleiSelection(ind=1)
            for name in AllNames:
                msk = nib.load(Directory + name + mode + '.nii.gz').get_data()            
                smallFuncs.saveImage( closeMask(msk > 0) , im.affine , im.header, Directory + 'ImClosed/' + name + '_ImClosed' + mode + '.nii.gz')

        def Save_AllNuclei_inOne():
            def edgeDetect(msk,SD):    
                print('---')            
                for i in range(msk.shape[2]): 
                    msk[...,i] = canny(msk[...,i] , low_threshold=0.1 , high_threshold=0.9)
                return msk

            for cnt , name in enumerate(smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei().Names):                
                msk = nib.load( Directory + 'ImClosed/' + name + '_ImClosed' + mode + '.nii.gz' ).get_data()  
                if cnt != 0:
                    Mask = (cnt+1)*msk if cnt == 1 else Mask + (cnt+1)*msk   
                else:
                    mskTh = msk.copy() # 
                
            for SD in range(3):
                a , b = sliceDim(SD)
                msk2 = mskTh.copy()
                msk2 = edgeDetect(msk2.transpose(a) , SD).transpose(b)
                smallFuncs.saveImage( Mask + msk2 , im.affine , im.header, Directory + 'AllLabels' + str(SD) + '.nii.gz')
        
        # saving4SuperNuclei()
        Save_AllNuclei_inOne()

        # saving4SuperNuclei_WithDifferentLabels()

        # creatingFullMaskWithAll4Supernuclei()

        # ImClosingAllNuclei()

    Subjects = [sub for sub in os.listdir(Dir) if 'vimp' in sub]

    for nameSubject in Subjects:
        print(nameSubject)        
        RunAllFunctions(Dir + '/' + nameSubject + '/Label/')


# params = paramFunc.Run(UserInfo.__dict__, terminal=True)
Dir = '/array/ssd/msmajdi/experiments/keras/exp1/test/Main/' # params.directories.Test.Input.Subjects  + '/' # 

mode = '_PProcessed'
applyMain(Dir ,mode)
# applyMain(Dir + '/train/Augments',mode)

# applyMain(Dir + '/test',mode)
