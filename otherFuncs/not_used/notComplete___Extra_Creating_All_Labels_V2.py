import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os, sys
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# import otherFuncs.smallFuncs as smallFuncs
from scipy import ndimage
from skimage.feature import canny
from mpl_toolkits import mplot3d
from skimage import measure

class Nuclei_Class():        
        
    def __init__(self, index=1, method = 'HCascade'):

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
                14: '14-MTT',
                1.1: 'lateral_ImClosed',
                1.2: 'posterior_ImClosed',
                1.3: 'Medial_ImClosed',
                1.4: 'Anterior_ImClosed',
                1.9: 'HierarchicalCascade' }  
            return switcher.get(index, 'wrong index')
        self.name = nucleus_name_func(index)

        self.nucleus_name_func = nucleus_name_func
        self.method      = method
        self.child       = ()
        self.parent      = None
        self.grandparent = None
        self.index = index
        

        def find_Parent_child(self):
            def parent_child(index):
                if self.method == 'HCascade':
                    switcher_Parent = {
                        1:   (None, [1.1 , 1.2 , 1.3 , 2]),
                        1.1: (1,    [4,5,6,7]),   # Lateral
                        1.2: (1,    [8,9,10]),    # Posterior
                        1.3: (1,    [11,12,13]),  # Medial
                        1.4: (1,     None),       # Anterior
                        2:   (1,     None) }              
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
        if self.method == 'HCascade': indexes = tuple([1,2,4,5,6,7,8,9,10,11,12,13,14]) + tuple([1.1,1.2,1.3])
        else:                         indexes = tuple([1,2,4,5,6,7,8,9,10,11,12,13,14])

        class All_Nuclei:
            Indexes = indexes[:]
            Names  = [self.nucleus_name_func(index) for index in Indexes]

        return All_Nuclei()    
    
    def HCascade_Parents_Identifier(self, Nuclei_List):
  
        def fchild(ix): return Nuclei_Class(ix , self.method).child
        return [ix for ix in fchild(1) if fchild(ix) and bool(set(Nuclei_List) & set(fchild(ix))) ]  if self.method == 'HCascade' else [1]
        
    def remove_Thalamus_From_List(self , Nuclei_List):
        nuLs = Nuclei_List.copy()
        if 1 in nuLs: nuLs.remove(1)
        return nuLs
            
def saveImage(Image , Affine , Header , outDirectory):
    def mkDir(Dir):
        if not os.path.isdir(Dir): os.makedirs(Dir)
        return Dir

    mkDir(outDirectory.split(os.path.basename(outDirectory))[0])
    out = nib.Nifti1Image((Image).astype('float32'),Affine)
    out.get_header = Header
    nib.save(out , outDirectory)

class UserEntry():
    def __init__(self):
        self.dir_in  = ''
        self.dir_out = ''
        self.mode    = 0

        for en in range(len(sys.argv)):
            if sys.argv[en].lower() in ('-i','--input'):    self.dir_in  = os.getcwd() + '/' + sys.argv[en+1]
            elif sys.argv[en].lower() in ('-m','--mode'):   self.mode    = int(sys.argv[en+1])                     
            
        print(self.dir_in)
        print(self.dir_out)


class merge_Labels():
    def __init__(self, dir_in = ''):
        self.dir_in  = dir_in

    def saving4SuperNuclei():
        print('    saving 4 Super Nuclei')
        for superNuclei in HierarchicalNames:
            for cnt, subNuclei in enumerate(Names[superNuclei].FullNames):
                msk = nib.load(Directory + subNuclei + mode + '.nii.gz').get_data()
                Mask = msk if cnt == 0 else Mask + msk

            smallFuncs.saveImage( Mask > 0 , im.affine , im.header, Directory + 'Hierarchical/' + superNuclei + mode + '.nii.gz')
            smallFuncs.saveImage( closeMask(Mask > 0 , 1) , im.affine , im.header, Directory + superNuclei + '_ImClosed' + mode + '.nii.gz')

    def saving4SuperNuclei_WithDifferentLabels():
        print('    saving 4 Super Nuclei')
        for superNuclei in HierarchicalNames:
            for cnt, subNuclei in enumerate(Names[superNuclei].FullNames):
                msk = nib.load(Directory + subNuclei + mode + '.nii.gz').get_data()
                Mask = msk if cnt == 0 else Mask + (cnt+1)*msk

            smallFuncs.saveImage( Mask , im.affine , im.header, Directory + superNuclei + mode + '_DifferentLabels.nii.gz')
            # smallFuncs.saveImage( closeMask(Mask , 1) , im.affine , im.header, Directory + superNuclei + '_ImClosed' + mode + '_DifferentLabels.nii.gz')

    def creatingFullMaskWithAll4Supernuclei():
        print('    creating Full Mask With All 4 Super Nuclei')
        for cnt, superNuclei in enumerate(HierarchicalNames):
            msk = nib.load(Directory + 'Hierarchical/' + superNuclei + mode + '.nii.gz').get_data()
            Mask = msk if cnt == 0 else Mask + (cnt+1)*msk

            msk = nib.load(Directory + superNuclei + '_ImClosed' + mode + '.nii.gz').get_data()
            MaskClosed = msk if cnt == 0 else MaskClosed + (cnt+1)*msk

        smallFuncs.saveImage(Mask , im.affine , im.header, Directory + 'Hierarchical/All_4MainNuclei' + mode + '.nii.gz')
        smallFuncs.saveImage(MaskClosed , im.affine , im.header, Directory + 'Hierarchical/All_4MainNuclei_ImClosed' + mode + '.nii.gz')

    def Save_AllNuclei_inOne():
        A = smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei()
        Mask = []
        for cnt , name in zip(A.Indexes , A.Names):                                
            if cnt != 1:
                msk = nib.load( Directory + 'ImClosed/' + name + '_ImClosed' + mode + '.nii.gz' ).get_data()  
                Mask = cnt*msk if Mask == [] else Mask + cnt*msk   

        smallFuncs.saveImage( Mask , im.affine , im.header, Directory + 'AllLabels.nii.gz')
        
    def loop_All_subjects(self):
        for subj in [s for s in os.listdir(self.dir_in) if 'vimp' in s]:
            print(subj , '\n')
            dir_in  = self.dir_in + '/' + subj
            dir_out = self.dir_out + '/' + subj
            temp = reslice_cls(dir_in=dir_in , dir_out=dir_out)
            temp.apply()



UI = UserEntry()
# UI.dir_in  = '/array/ssd/msmajdi/experiments/keras/exp4/test/Main/vimp2_case2'
# UI.dir_out = '/array/ssd/msmajdi/experiments/keras/exp4/test/Main/vimp2_case2_Reslice3'
# UI.mode = 0

if UI.mode == 0: merge_Labels(dir_in = UI.dir_in).apply()
else:            merge_Labels(dir_in = UI.dir_in).loop_All_subjects()
