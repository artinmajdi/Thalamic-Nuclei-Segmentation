import os
import nibabel as nib
from paramsFunctions import checkTestDirectory , fixDirectoryLastDashSign , funcExpDirectories
import numpy as np
import sys


# a = funcExpDirectories(1,1)
# b = a()

def NucleiSelection(ind):

    if ind == 1:
        NucleusName = '1-THALAMUS'
    elif ind == 2:
        NucleusName = '2-AV'
    elif ind == 4567:
        NucleusName = '4567-VL'
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

    return NucleusName

def listSubFolders(Dir_Prior):

    oldStandard = 1
    if oldStandard == 1:
        subFolders = []
        subFlds = os.listdir(Dir_Prior)
        for i in range(len(subFlds)):
            if subFlds[i][:5] == 'vimp2':
                subFolders.append(subFlds[i])
    else:
        subFolders = os.listdir(Dir_Prior)

    return subFolders

def mkDir(dir):
    try:
        os.stat(dir)
    except:
        os.makedirs(dir)
    return dir

def choosingSubject(Input):
    return Input.Image.get_data() , Input.CropMask.get_data() , Input.ThalamusMask.get_data() , Input.TestAddress

def saveImage(Image , Affine , Header , outDirectory):
    out = nib.Nifti1Image(Image,Affine)
    out.get_header = Header
    nib.save(out , outDirectory)

def saveMain(Input):

    outDirectory = ''
    for AugIx in Input.Image.shape[3]:
        saveImage(Input.Image[...,AugIx] , Input.Affine , Input.Header , outDirectory)
        saveImage(Input.CropMask[...,AugIx] , Input.Affine , Input.Header , outDirectory)
        saveImage(Input.ThalamusMask[...,AugIx] , Input.Affine , Input.Header , outDirectory)

def terminalEntries(params):

    params.gpuNum =  '4'  # 'nan'  #
    params.IxNuclei = [1]
    params.whichMachine = 'server'

    for en in range(len(sys.argv)):
        entry = sys.argv[en]

        if entry.lower() == '-g':
            params.gpuNum = sys.argv[en+1]

        elif entry.lower() == '-i':
            params.directories.Input , params.MultipleTest = checkTestDirectory( sys.argv[en+1] )

        elif entry.lower() == '-o':
            params.directories.Output = sys.argv[en+1]

        elif entry.lower() == '-m':
            params.whichMachine = sys.argv[en+1]

        elif entry.lower() == '-n':
            if sys.argv[en+1].lower() == 'all':
                params.IxNuclei = np.append([1,2,4567],range(4,14))

            elif sys.argv[en+1][0] == '[':
                B = sys.argv[en+1].split('[')[1].split(']')[0].split(",")
                params.IxNuclei = [int(k) for k in B]

            else:
                params.IxNuclei = [int(sys.argv[en+1])]

    return params
