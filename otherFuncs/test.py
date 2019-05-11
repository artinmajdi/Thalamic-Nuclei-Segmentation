# import numpy as np
from sklearn import tree
import os
import nibabel as nib
import numpy as np
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc

params = paramFunc.Run(UserInfoB, terminal=True)

dir = 'dir to results'
a = [s for s in os.listdir(dir + 'sd0/') if 'vimp' in s]
b = [s for s in os.listdir(dir + 'sd1/') if 'vimp' in s]
c = [s for s in os.listdir(dir + 'sd2/') if 'vimp' in s]
subjectList = list(set(a+b+c))


def load_msks_Allsd(dir + sd + subj + '/' +  label):
    def readAllsd(sd):
        msk = nib.load(dir + sd + subj + '/' +  label + '.nii.gz')
        return msk.get_data().reshape(-1).transpose()

    X = [readAllsd(sd) for sd in ['sd0/' , 'sd1/' , 'sd2/'] ]
    


for subj in params.directories.Input.subjects:
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,Y)



dir = '/home/artinl/Documents/RESEARCH/dataset/Main/vimp2_819_05172013_DS_MS/Label/1-THALAMUS_PProcessed.nii.gz'
msk = nib.load(dir)

a.shape

X = []
X.append(a)
X
b = a.reshape(msk.shape)
b.shape

np.unique(msk.get_data())
clf.predict(X2)
