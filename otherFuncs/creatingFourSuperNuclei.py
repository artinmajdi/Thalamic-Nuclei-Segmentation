import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os, sys
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from otherFuncs import smallFuncs
from scipy import ndimage


def applyMain(Dir,mode):
    
    def RunAllFunctions(Directory):
        im = nib.load(Directory + '1-THALAMUS' + mode + '.nii.gz')
        Names = dict({1:['posterior',[8,9,10,11]], 2:['lateral',[4,5,6,7]] , 3:['Anterior',[2]] , 4:['Medial',[12]] })

        def closeMask(mask):
            struc = ndimage.generate_binary_structure(3,2)
            return ndimage.binary_closing(mask, structure=struc)

        def saving4SuperNuclei():
            print('    saving 4 Super Nuclei')
            for mIx in range(1,5):
                for cnt, nix in enumerate(Names[mIx][1]):
                    name, _, _ = smallFuncs.NucleiSelection(ind=nix,organ='THALAMUS')
                    msk = nib.load(Directory + name + mode + '.nii.gz').get_data()
                    Mask = msk if cnt == 0 else Mask + msk

                smallFuncs.saveImage( Mask > 0 , im.affine , im.header, Directory + 'Hierarchical/' + Names[mIx][0] + mode + '.nii.gz')
                smallFuncs.saveImage( closeMask(Mask > 0) , im.affine , im.header, Directory + 'Hierarchical/' + Names[mIx][0] + '_ImClosed' + mode + '.nii.gz')

        def creatingFullMaskWithAll4Supernuclei():
            print('    creating Full Mask With All 4 Super Nuclei')
            for cnt in range(1,5):
                msk = nib.load(Directory + 'Hierarchical/' + Names[cnt][0] + mode + '.nii.gz').get_data()
                Mask = msk if cnt == 1 else Mask + cnt*msk

                msk = nib.load(Directory + 'Hierarchical/' + Names[cnt][0] + '_ImClosed' + mode + '.nii.gz').get_data()
                MaskClosed = msk if cnt == 1 else MaskClosed + cnt*msk

            smallFuncs.saveImage(Mask , im.affine , im.header, Directory + 'Hierarchical/All_4MainNuclei' + mode + '.nii.gz')
            smallFuncs.saveImage(MaskClosed , im.affine , im.header, Directory + 'Hierarchical/All_4MainNuclei_ImClosed' + mode + '.nii.gz')

        def ImClosingAllNuclei():
            print('    ImClosing All Nuclei')
            _, _, AllNames = smallFuncs.NucleiSelection(ind=1,organ='THALAMUS')
            for name in AllNames:
                msk = nib.load(Directory + name + mode + '.nii.gz').get_data()            
                smallFuncs.saveImage( closeMask(msk > 0) , im.affine , im.header, Directory + 'ImClosed/' + name + '_ImClosed' + mode + '.nii.gz')
                
        saving4SuperNuclei()

        creatingFullMaskWithAll4Supernuclei()

        ImClosingAllNuclei()

    Subjects = [sub for sub in os.listdir(Dir) if 'vimp2' in sub]

    for nameSubject in Subjects:
        print(nameSubject)        
        RunAllFunctions(Dir + '/' + nameSubject + '/Label/')



Dir = '/array/ssd/msmajdi/data/preProcessed/7T/All'

mode = '_PProcessed'
# applyMain(Dir + '/train',mode)
# applyMain(Dir + '/test',mode)
if os.path.exists(Dir + '/train/Augments'): applyMain(Dir + '/train/Augments',mode)
