import os

def RigidRegistration(directories):
    # os.system("ANTS 3 -m CC\[%s/*_bias_corr.nii.gz, %s ,1,5\] -o linear -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(directories.Input ,directories.TemplateImage) )
    # os.system("WarpImageMultiTransform 3 %s MyCrop2_Gap20.nii.gz -R %s/*_bias_corr.nii.gz linearAffine.txt"%(directories.TemplateMask ,directories.Input) )

    os.system("ANTS 3 -m CC[%s/*_bias_corr.nii.gz, %s ,1,5] -o linear -i 0 --use-Histogram-Matching --number-of-affine-iterations 10000x10000x10000x10000x10000 --MI-option 32x16000 --rigid-affine false" %(directories.Input ,directories.TemplateImage) )
    os.system("WarpImageMultiTransform 3 %s MyCrop2_Gap20.nii.gz -R %s/*_bias_corr.nii.gz linearAffine.txt"%(directories.TemplateMask ,directories.Input) )

def BiasCorrection(dir,name):
    try:
        os.stat( dir + '/' + name + '_bias_corr.nii.gz' )
    except:
        os.system("N4BiasFieldCorrection -d 3 -i %s/%s.nii.gz -o %s/%s_bias_corr.nii.gz -b [200] -s 3 -c [50x50x30x20,1e-6]"%(dir, name, dir, name)  )

def Cropping(params):
    # ExtractRegionFromImageByMask 3 input/*.nii.gz input/*_Cropped.nii.gz mask.nii.gz 1 0
    return True

def NonLinearWarp(params):
    return True
