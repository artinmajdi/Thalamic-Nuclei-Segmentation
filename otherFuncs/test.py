import numpy as np
import nibabel as nib
import os


dir_Train = os.listdir('/array/ssd/msmajdi/data/preProcessed/7T/All_DBD/train')

dir = '/array/ssd/msmajdi/data/preProcessed/7T/All_DBD/test'

for sb in os.listdir(dir):
    # if sb in dir_Train:
    if 'Test_' in sb:

    
    # os.system('mv %s/%s/vtk_rois %s/%s/Label'%(dir,sb,dir,sb))
        # sb2 = sb.split('vimp_')[1] if 'vimp_' in sb else sb.split('vimp2_')[1]
        sb2 = sb.split('Test_')[1] #  if '_MS' in sb else sb.split('_priors_control')[0]
        os.system('mv %s/%s %s/%s'%(dir,sb,dir,sb2))
