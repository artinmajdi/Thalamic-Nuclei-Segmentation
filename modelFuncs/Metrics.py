import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt



def MetricInfo(Metric_Index):
    switcher = {
        1: (mDice, 'Dice'),
        2: ('acc', 'Acc'),
        3: (['acc',mDice], 'Acc_N_Dice')
    }
    return switcher.get(Metric_Index, 'WARNING: Invalid metric index')


def mDice(y_true,y_pred):

    Dice = 0    
    nmCl = max(y_pred.shape[3] - 1,1)  # max is for when I don't add the background to the label concatenate(msk,1-msk)
    for d in range(nmCl):
        Dice = Dice + tf.reduce_sum(tf.multiply(y_true[...,d],y_pred[...,d]))*2/( tf.reduce_sum(y_true[...,d]) + tf.reduce_sum(y_pred[...,d]) + 1e-7)
    
    return tf.divide(Dice,tf.cast(nmCl,tf.float32))


def JAC(y_true,y_pred):
   
    Intersection = tf.reduce_sum(tf.multiply(y_true,y_pred))
    Sum = ( tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)
    return Intersection(Sum - Intersection) 


def VSI(y_true,y_pred):
    def main_VSI(yy, xx):
        X = tf.reduce_sum(xx)
        Y = tf.reduce_sum(yy)    
        return 1 - ( tf.abs(X-Y) / (X+Y) )

    nmCl = max(y_pred.shape[3] - 1,1)  
    return tf.reduce_sum( [main_VSI(y_true[...,d], y_pred[...,d]) for d in range(nmCl)] )


def confusionMatrix(y_true,y_pred):
    
    yp1 = np.reshape(y_pred,[-1,1])
    yt1 = np.reshape(y_true,[-1,1])

    D = confusion_matrix(yt1, yp1 > 0.5)

    class metrics():
        TN, FP, FN , TP = D[0,0] , D[0,1] , D[1,0] , D[1,1]
        Recall = TP / (TP + FN)
        Sensitivity = TP / (TP + FN)
        Specificity = TN/(TN + FP)
        Precision   = TP/(TP + FP)

    return metrics()

def ROC_Curve(y_true,y_pred):
    yp1 = np.reshape(y_pred,[-1,1])
    yt1 = np.reshape(y_true,[-1,1])

    fpr, tpr, _ = roc_curve(yt1, yp1,pos_label=[0])
    auc(fpr, tpr)
    # plt.plot(fpr, tpr)

def Precision_Recall_Curve(y_true,y_pred):

    yp1 = np.reshape(y_pred,[-1,1])
    yt1 = np.reshape(y_true,[-1,1])

    precision, recall, thresholds = precision_recall_curve(yt1, yp1)
    average_precision = average_precision_score(yt1, yp1)

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision))
