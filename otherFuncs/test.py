import keras.models as kmodel
import sys, os
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# import keras.layers as KLayers
# from tqdm import tqdm
import numpy as np
# from scipy.spatial import distance
import nibabel as nib
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

subject = 'vimp2_0781_05292014_SRI/'
# dir = '/home/artinl/Documents/vimp2_0699_04302014_SRI/'
dir = '/array/ssd/msmajdi/experiments/keras/exp6/results/sE8_Cascade_FM10_Res_Unet2_NL3_LS_MyLogDice_US1_SingleClass/sd0/' + subject
y_pred = nib.load(dir+'2-AV.nii.gz').get_data()

# dir = '/home/artinl/Documents/vimp2_0699_04302014_SRI/Label/'
dir = '/array/ssd/msmajdi/experiments/keras/exp6/test/SRI/' + subject + 'Label/'
y_true = nib.load(dir+'2-AV_PProcessed.nii.gz').get_data()


def mDice(msk1,msk2):
    intersection = msk1*msk2
    return intersection.sum()*2/(msk1.sum()+msk2.sum() + np.finfo(float).eps)


yp1 = np.reshape(y_pred,[-1,1])
yt1 = np.reshape(y_true,[-1,1])

# mDice(y_pred > 0.5,y_true)

D = confusion_matrix(yt1, yp1 > 0.5)
TN, FP, FN , TP = D[0,0] , D[0,1] , D[1,0] , D[1,1]

Recall = Sensitivity = TP / (TP + FN)
Specificity = TN/(TN + FP)
Precision = TP/(TP + FP)


from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

fpr, tpr, _ = roc_curve(yt1, yp1,pos_label=[0])
auc(fpr, tpr)
# plt.plot(fpr, tpr)

precision, recall, thresholds = precision_recall_curve(yt1, yp1)
average_precision = average_precision_score(yt1, yp1)

# f1 = f1_score(precision, recall)

plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision))
