import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os, sys
import pathlib
from scipy import ndimage
from skimage.feature import canny
from mpl_toolkits import mplot3d
from skimage import measure

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from Parameters import UserInfo, paramFunc
from otherFuncs import smallFuncs


params = paramFunc.Run(UserInfo.__dict__, terminal=True)

def sliceDim(SD):
    if   SD == 0: return ([1,2,0] , [2,0,1])
    elif SD == 1: return ([2,0,1] , [1,2,0])
    elif SD == 2: return ([0,1,2] , [0,1,2])
    
def Save_AllNuclei_inOne(Directory,mode):
    A = smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei()
    Mask = []
    for cnt , name in zip(A.Indexes , A.Names):                                
        if cnt != 1:
            msk = nib.load( Directory + '/' + name + mode + '.nii.gz' ).get_fdata()  
            Mask = cnt*msk if Mask == [] else Mask + cnt*msk 
        else:
            im = nib.load( Directory + '/' + name + mode + '.nii.gz' )  

    smallFuncs.saveImage( Mask , im.affine , im.header, Directory + '/AllLabels.nii.gz')


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
        
        def closeMask(mask,cnt):
            struc = ndimage.generate_binary_structure(3,2)
            if cnt > 1: struc = ndimage.iterate_structure(struc, cnt)
            return ndimage.binary_closing(mask, structure=struc)  
            
        def saving4SuperNuclei():
            print('    saving 4 Super Nuclei')
            for superNuclei in HierarchicalNames:
                for cnt, subNuclei in enumerate(Names[superNuclei].FullNames):
                    msk = nib.load(Directory + subNuclei + mode + '.nii.gz').get_fdata()
                    Mask = msk if cnt == 0 else Mask + msk

                smallFuncs.saveImage( Mask > 0 , im.affine , im.header, Directory + 'Hierarchical/' + superNuclei + mode + '.nii.gz')
                smallFuncs.saveImage( closeMask(Mask > 0 , 1) , im.affine , im.header, Directory + superNuclei + '_ImClosed' + mode + '.nii.gz')

        def saving4SuperNuclei_WithDifferentLabels():
            print('    saving 4 Super Nuclei')
            for superNuclei in HierarchicalNames:
                for cnt, subNuclei in enumerate(Names[superNuclei].FullNames):
                    msk = nib.load(Directory + subNuclei + mode + '.nii.gz').get_fdata()
                    Mask = msk if cnt == 0 else Mask + (cnt+1)*msk

                smallFuncs.saveImage( Mask , im.affine , im.header, Directory + superNuclei + mode + '_DifferentLabels.nii.gz')

        def creatingFullMaskWithAll4Supernuclei():
            print('    creating Full Mask With All 4 Super Nuclei')
            for cnt, superNuclei in enumerate(HierarchicalNames):
                msk = nib.load(Directory + 'Hierarchical/' + superNuclei + mode + '.nii.gz').get_fdata()
                Mask = msk if cnt == 0 else Mask + (cnt+1)*msk

                msk = nib.load(Directory + superNuclei + '_ImClosed' + mode + '.nii.gz').get_fdata()
                MaskClosed = msk if cnt == 0 else MaskClosed + (cnt+1)*msk

            smallFuncs.saveImage(Mask , im.affine , im.header, Directory + 'Hierarchical/All_4MainNuclei' + mode + '.nii.gz')
            smallFuncs.saveImage(MaskClosed , im.affine , im.header, Directory + 'Hierarchical/All_4MainNuclei_ImClosed' + mode + '.nii.gz')

        def ImClosingAllNuclei():
            print('    ImClosing All Nuclei')
            _, _, AllNames = smallFuncs.NucleiSelection(ind=1)
            for name in AllNames:
                msk = nib.load(Directory + name + mode + '.nii.gz').get_fdata()            
                smallFuncs.saveImage( closeMask(msk > 0 , 1) , im.affine , im.header, Directory + 'ImClosed/' + name + '_ImClosed' + mode + '.nii.gz')

        def edgeDetect(msk,SD):    
            print('---')            
            for i in range(msk.shape[2]): 
                msk[...,i] = canny(msk[...,i] , low_threshold=0.1 , high_threshold=0.9)
            return msk

        def Save_AllNuclei_inOne_Imclosed_Except_AV():
            A = smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei()
            Mask = []
            for cnt , name in zip(A.Indexes , A.Names):                                
                if cnt not in [1, 2]:                    
                    msk = nib.load( Directory + name + mode + '.nii.gz' ).get_fdata()  
                    msk = closeMask(msk > 0 , 1)
                    Mask = msk if Mask == [] else Mask + msk   

            smallFuncs.saveImage( closeMask(Mask,2) > 0.5, im.affine , im.header, Directory + 'AllLabels_Except_AV.nii.gz')

        def Save_AllNuclei_inOne():
            A = smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei()
            Mask = []
            for cnt , name in zip(A.Indexes , A.Names):                                
                if cnt != 1:
                    msk = nib.load( Directory + 'ImClosed/' + name + '_ImClosed' + mode + '.nii.gz' ).get_fdata()  
                    Mask = cnt*msk if Mask == [] else Mask + cnt*msk   

            smallFuncs.saveImage( Mask , im.affine , im.header, Directory + 'AllLabels.nii.gz')
        
        saving4SuperNuclei()

        ImClosingAllNuclei()

        Save_AllNuclei_inOne()
        # Save_AllNuclei_inOne_Imclosed_Except_AV()
        saving4SuperNuclei_WithDifferentLabels()
        # saveAV_BB()
        creatingFullMaskWithAll4Supernuclei()

    Subjects = [sub for sub in os.listdir(Dir) if 'case' in sub]

    for nameSubject in Subjects:
        print(nameSubject , Dir)        
        RunAllFunctions(Dir + nameSubject + '/Label/')


class Input_cls():
    def __init__(self, dir_in='' , dir_out=''):
        
        self.dir_in = dir_in
        self.dir_out = dir_out

        def directories(self):
            for en in range(len(sys.argv)):

                if sys.argv[en].lower() in ('-i','--input'):    self.dir_in = sys.argv[en+1]
                elif sys.argv[en].lower() in ('-o','--output'): self.dir_out = sys.argv[en+1]
                           
        directories(self)

        if self.dir_in: self.subjList = [s for s  in os.listdir(self.dir_in) if ('case' in s) and ('jpg' not in s)]


input = Input_cls()

Save_AllNuclei_inOne(input.dir_in,'') 



