import os, sys
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
sys.path.append( os.path.dirname(os.path.dirname(__file__)) )
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
from nilearn import image as imageNilearn
import otherFuncs.smallFuncs as smallFuncs
import nibabel as nib

UserInfoB = smallFuncs.terminalEntries(UserInfo.__dict__)

UserInfoB['readAugments'].Mode = True
UserInfoB['ReadTrain'].SRI     = True
UserInfoB['ReadTrain'].ET      = True
UserInfoB['ReadTrain'].Main    = True
params = paramFunc.Run(UserInfoB, terminal=False)


def loopOver_AllSubjects(subjects):
    def func_upsample(inputAdr, name, scale):
        outputAdr = smallFuncs.mkDir(inputAdr.replace('exp7_cascadeV1' , 'exp8_cascadeV1'))
        im = imageNilearn.load_img(inputAdr + '/' + name)

        affine = im.affine.copy()
        for i in range(3): affine[i,i] /= scale

        im_US = imageNilearn.resample_img(img=im , target_affine=affine,interpolation='nearest')
        nib.save(im_US, outputAdr + '/' + name)
            
    for ix, (_,subj) in enumerate(subjects.items()):

        print(str(ix) + '/' + str(len(subjects)), subj.address.split('exp7_cascadeV1')[1])
        images = [s for s in os.listdir(subj.address) if 'PProcessed.nii.gz' in s]
        for im in images: func_upsample(subj.address, im , 2)
        
        # print('    ', 'CropMask')
        # func_upsample(subj.address + '/temp', 'CropMask.nii.gz' , 2)
        
        labels = [s for s in os.listdir(subj.Label.address) if 'PProcessed.nii.gz' in s]
        for msk in labels: func_upsample(subj.Label.address, msk , 2)

loopOver_AllSubjects(params.directories.Train.Input.Subjects)
loopOver_AllSubjects(params.directories.Test.Input.Subjects)