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
from mpl_toolkits import mplot3d
from skimage import measure

params = paramFunc.Run(UserInfo.__dict__, terminal=True)

def sliceDim(SD):
    if   SD == 0: return ([1,2,0] , [2,0,1])
    elif SD == 1: return ([2,0,1] , [1,2,0])
    elif SD == 2: return ([0,1,2] , [0,1,2])
    

def Save_AllNuclei_inOne(Directory):
    A = smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei()
    Mask = []
    for cnt , name in zip(A.Indexes , A.Names):                                
        if cnt != 1:
            msk = nib.load( Directory + '/' + name + '.nii.gz' ).get_data()  
            Mask = cnt*msk if Mask == [] else Mask + cnt*msk 
        else:
            im = nib.load( Directory + '/' + name + '.nii.gz' )  

    smallFuncs.saveImage( Mask , im.affine , im.header, Directory + 'AllLabels.nii.gz')
    
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

        def dilateMask(mask,cnt):
            struc = ndimage.generate_binary_structure(3,2)
            if cnt > 1: struc = ndimage.iterate_structure(struc, cnt)
            return ndimage.binary_dilation(mask, structure=struc)
        
        def closeMask(mask,cnt):
            struc = ndimage.generate_binary_structure(3,2)
            if cnt > 1: struc = ndimage.iterate_structure(struc, cnt)
            return ndimage.binary_closing(mask, structure=struc)  
            
        def saving4SuperNuclei():
            print('    saving 4 Super Nuclei')
        
            for superNuclei in HierarchicalNames:
                if not os.path.exists(Directory + superNuclei + '_ImClosed' + mode + '.nii.gz'):
                    for cnt, subNuclei in enumerate(Names[superNuclei].FullNames):
                        msk = nib.load(Directory + subNuclei + mode + '.nii.gz').get_data()
                        Mask = msk if cnt == 0 else Mask + msk

                    smallFuncs.saveImage( Mask > 0 , im.affine , im.header, Directory + 'Hierarchical/' + superNuclei + mode + '.nii.gz')
                    smallFuncs.saveImage( closeMask(Mask > 0 , 1) , im.affine , im.header, Directory + superNuclei + '_ImClosed' + mode + '.nii.gz')

        def saving4SuperNuclei_WithDifferentLabels():
            print('    saving 4 Super Nuclei with different labels')
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
                # a = mplot3d.Axes3D.contour3D(x,y,z)
                # mplot3d.Axes3D.scatter3D(x2,y2,zs=1)
                # mplot3d.Axes3D(x2,y2)
                # x,y,z = np.where(msk == 1)
                # x2,y2 = np.meshgrid(x,y)

                # from mpl_toolkits.mplot3d import axes3d
                # import matplotlib.pyplot as plt
                # from matplotlib import cm

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # X, Y, Z = axes3d.get_test_data(0.05)
                # cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
                # ax.clabel(cset, fontsize=9, inline=1)
                # plt.show()
                msk = nib.load(Directory + superNuclei + '_ImClosed' + mode + '.nii.gz').get_data()
                MaskClosed = msk if cnt == 0 else MaskClosed + (cnt+1)*msk

            smallFuncs.saveImage(Mask , im.affine , im.header, Directory + 'Hierarchical/All_4MainNuclei' + mode + '.nii.gz')
            smallFuncs.saveImage(MaskClosed , im.affine , im.header, Directory + 'Hierarchical/All_4MainNuclei_ImClosed' + mode + '.nii.gz')

        def saveAV_BB():             
            print('    creating Full Mask With All 4 Super Nuclei')
            for cnt, superNuclei in enumerate([HierarchicalNames[0],HierarchicalNames[2]]):
                msk = nib.load(Directory + superNuclei + '_ImClosed' + mode + '.nii.gz').get_data()
                Mask_Lateral_Medial = msk if cnt == 0 else Mask_Lateral_Medial + msk

            BBf = np.loadtxt('/array/ssd/msmajdi/experiments/keras/exp1/results/sE11_HCascade_FM20_7T_Main/sd2/vimp2_915_07112013_LC_MS/BB_1-THALAMUS.txt',dtype=int)
            crd = BBf[:,:2]

            mskTh = nib.load(Directory + '1-THALAMUS_PProcessed.nii.gz').get_data()[crd[0,0]:crd[0,1] , crd[1,0]:crd[1,1] , crd[2,0]:crd[2,1]]
            mskAV = nib.load(Directory + '2-AV_PProcessed.nii.gz').get_data()[crd[0,0]:crd[0,1] , crd[1,0]:crd[1,1] , crd[2,0]:crd[2,1]]
            mskAll = nib.load(Directory + 'AllLabels2.nii.gz').get_data()[crd[0,0]:crd[0,1] , crd[1,0]:crd[1,1] , crd[2,0]:crd[2,1]]
            
            # def remove_posterior(im, Directory):
            mskPosterior = nib.load(Directory + 'posterior_ImClosed_PProcessed.nii.gz').get_data()[crd[0,0]:crd[0,1] , crd[1,0]:crd[1,1] , crd[2,0]:crd[2,1]]
            objects = measure.regionprops(measure.label(mskPosterior))
            Ix = np.argsort( [obj.area for obj in objects] )
            bbox = objects[ Ix[-1] ].bbox
            crd = [ [bbox[d] , bbox[3 + d] ] for d in range(3)]
            coronalCrop = [ crd[1][1] , mskPosterior.shape[1] ] 


            print('---------')
                # return im[crd[0,0]:crd[0,1] , crd[1,0]:crd[1,1] , crd[2,0]:crd[2,1]]

            # mskPosterior2 = dilateMask(mskPosterior,3) > 0.5
            # mskPosterior2[mskTh == False] = False

            # mskTh_WoPosterior = mskTh.copy()
            # mskTh_WoPosterior[np.where(mskPosterior2 == 1)] = 0
            # mskTh_WoPosterior[mskTh == False] = False

            # ThEdge = edgeDetect(mskTh,2)
            # Mask_Lateral_Medial_Closed = (closeMask(Mask_Lateral_Medial + ThEdge,1) > 0.5)
            # Mask_Lateral_Medial_Closed[mskPosterior2 == True] = False
            # Mask_Lateral_Medial_Closed[ThEdge > 0.5] = False
            # # Mask_Lateral_Medial_Closed = dilateMask(Mask_Lateral_Medial_Closed,2) > 0.5

            # mskTh_Anterior = mskTh.copy()
            # mskTh_Anterior[Mask_Lateral_Medial_Closed] = False
            # mskTh_Anterior[mskPosterior2] = False

            # mskTh_Anterior2 = 15*mskTh
            # mskTh_Anterior2[mskAV>0.5] = 5
            # msk  = (mskTh - (Mask > 0.5))
            
            # mskFinal = mskTh-Mask2
            # mskFinal[np.where(mskPosterior2 == 1)] = 0

            # a = nib.viewers.OrthoSlicer3D(Mask_Lateral_Medial_Closed,title='Mask_Lateral_Medial_Closed')
            # b = nib.viewers.OrthoSlicer3D(mskTh_Anterior,title='AV')
            # c = nib.viewers.OrthoSlicer3D(mskAll,title='all nuclei')
            # a.link_to(b)
            # a.link_to(c)
            # a.show()

            smallFuncs.saveImage(msk , im.affine , im.header, Directory + 'AnteiorMask_2AV_new.nii.gz')

        def ImClosingAllNuclei():
            print('    ImClosing All Nuclei')
            _, _, AllNames = smallFuncs.NucleiSelection(ind=1)
            for name in AllNames:
                msk = nib.load(Directory + name + mode + '.nii.gz').get_data()            
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
                    msk = nib.load( Directory + name + mode + '.nii.gz' ).get_data()  
                    msk = closeMask(msk > 0 , 1)
                    Mask = msk if Mask == [] else Mask + msk   

            # mskTh = nib.load( Directory + '1-THALAMUS_PProcessed.nii.gz' ).get_data() 
            # mskAV = nib.load( Directory + '2-AV_PProcessed.nii.gz' ).get_data()        
            # Mask_AllExcept_AV = closeMask(Mask,2) 
            # mskFull = np.zeros(mskTh.shape)
            # # mskFull = mskTh.copy()
            # mskFull[mskTh > 0.5] = 0.3
            # mskFull[Mask_AllExcept_AV > 0.5] = 0.6
            # mskFull[mskAV > 0.5] = 0.9
            # nib.viewers.OrthoSlicer3D(mskFull , title='0.9:AV  0.6:rest  0.3:Thalmaus').show()

            smallFuncs.saveImage( closeMask(Mask,2) > 0.5, im.affine , im.header, Directory + 'AllLabels_Except_AV.nii.gz')

        def Save_AllNuclei_inOne():
            A = smallFuncs.Nuclei_Class(method='Cascade').All_Nuclei()
            Mask = []
            for cnt , name in zip(A.Indexes , A.Names):                                
                if cnt != 1:
                    msk = nib.load( Directory + 'ImClosed/' + name + '_ImClosed' + mode + '.nii.gz' ).get_data()  
                    Mask = cnt*msk if Mask == [] else Mask + cnt*msk   

            smallFuncs.saveImage( Mask , im.affine , im.header, Directory + 'AllLabels.nii.gz')
        
        saving4SuperNuclei()

        # ImClosingAllNuclei()

        # Save_AllNuclei_inOne()
        # Save_AllNuclei_inOne_Imclosed_Except_AV()
        saving4SuperNuclei_WithDifferentLabels()
        # saveAV_BB()
        creatingFullMaskWithAll4Supernuclei()

    Subjects = [sub for sub in os.listdir(Dir) if 'vimp' in sub]

    for nameSubject in Subjects:
        print(nameSubject , Dir)        
        RunAllFunctions(Dir + nameSubject + '/Label/')


class Input_cls():
    def __init__(self, dir_in='' , dir_out=''):
        
        self.dir_in = dir_in
        self.dir_out = dir_out

        def directories(self):
            for en in range(len(sys.argv)):

                if sys.argv[en].lower() in ('-i','--input'):    self.dir_in  = os.getcwd() + '/' + sys.argv[en+1] if '/array/ssd' not in sys.argv[en+1] else sys.argv[en+1]
                elif sys.argv[en].lower() in ('-o','--output'): self.dir_out = os.getcwd() + '/' + sys.argv[en+1] if '/array/ssd' not in sys.argv[en+1] else sys.argv[en+1]
                           
        directories(self)

        self.subjList = [s for s  in os.listdir(self.dir_in) if ('vimp' in s) and ('jpg' not in s)]


# Dir = '/array/ssd/msmajdi/experiments/keras/exp4/results/sE11_Cascade_FM20_DO0.3_ET_Init_Main_AllAugs_CV_a/sd2'
# for exp in ['exp5_CSFn']:
#     for ds in ['CSFn']:
#         print('\n\n\n  ' + exp + ' CSFn \n\n\n')
#         # for dataset in ['Main/' , 'SRI/' , 'ET/']:

#         Dir_Train = '/array/ssd/msmajdi/experiments/keras/' + exp + '/train/' + ds + '/'
#         applyMain(Dir_Train ,'_PProcessed')
#         for sd in ['sd0/' , 'sd1/' , 'sd2/']: applyMain(Dir_Train + 'Augments/' + sd ,'_PProcessed')


#         # Dir_Test = '/array/ssd/msmajdi/experiments/keras/' + exp + '/test/' + ds + '/'
#         # applyMain(Dir_Test ,'_PProcessed')
#         # for sd in ['sd0/' , 'sd1/' , 'sd2/']: applyMain(Dir_Test + 'Augments/' + sd ,'_PProcessed')


#         for cv in ['/a/' , '/b/' , '/c/']:
#             Dir_crossVal = '/array/ssd/msmajdi/experiments/keras/' + exp + '/crossVal/' + ds + cv
#             applyMain(Dir_crossVal ,'_PProcessed')
#             for sd in ['sd0/' , 'sd1/' , 'sd2/']: applyMain(Dir_crossVal + 'Augments/' + sd ,'_PProcessed')


input = Input_cls()
applyMain(input.dir_in ,'')


# applyMain(Dir ,'_PProcessed')


