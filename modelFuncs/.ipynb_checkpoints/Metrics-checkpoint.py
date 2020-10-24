import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff


def MetricInfo(Metric_Index):
    switcher = {
        1: (mDice, 'DICE'),
        2: ('acc', 'Acc'),
        3: (['acc',mDice], 'Acc_N_Dice')
    }
    return switcher.get(Metric_Index, 'WARNING: Invalid metric index')


def mDice(y_true,y_pred):

    Dice = 0    
    nmCl = max(y_pred.shape[3] - 1,1)  # max is for when I don't add the background to the label concatenate(msk,1-msk)
    for d in range(nmCl):
        Dice = Dice + tf.reduce_sum(input_tensor=tf.multiply(y_true[...,d],y_pred[...,d]))*2/( tf.reduce_sum(input_tensor=y_true[...,d]) + tf.reduce_sum(input_tensor=y_pred[...,d]) + 1e-7)
    
    return tf.divide(Dice,tf.cast(nmCl,tf.float32))


def JAC(y_true,y_pred):
   
    Intersection = tf.reduce_sum(input_tensor=tf.multiply(y_true,y_pred))
    Sum = ( tf.reduce_sum(input_tensor=y_true) + tf.reduce_sum(input_tensor=y_pred) + 1e-7)
    return Intersection(Sum - Intersection) 



    
class VSI_AllClasses_TF:
    def __init__(self, y_true,y_pred):
        self.true = y_true
        self.pred = y_pred

    def VSI(self):
        X = tf.reduce_sum(input_tensor=self.true)
        Y = tf.reduce_sum(input_tensor=self.pred)    
        return 1 - ( tf.abs(X-Y) / (X+Y) )
        
    def apply_to_all_classes(self):
        nmCl = max(self.pred.shape[3] - 1,1)  
        return tf.reduce_sum( input_tensor=[VSI_AllClasses(self.true[...,d], self.pred[...,d]).VSI() for d in range(nmCl)] )

class VSI_AllClasses:
    def __init__(self, y_true,y_pred):
        self.true = y_true
        self.pred = y_pred

    def VSI(self):
        X = self.true.sum()
        Y = self.pred.sum()   
        return 1 - ( np.abs(X-Y) / (X+Y) )
        
    def apply_to_all_classes(self):
        nmCl = max(self.pred.shape[3] - 1,1)  
        return [VSI_AllClasses(self.true[...,d], self.pred[...,d]).VSI() for d in range(nmCl)].mean()

class HD_AllClasses:
    def __init__(self, y_true,y_pred):
        self.true = y_true
        self.pred = y_pred

    def HD(self):
        a = np.zeros((self.true.shape[2],2))
        for i in range(self.true.shape[2]): a[i,:] = [self.true[...,i].sum(), directed_hausdorff(self.true[...,i],self.pred[...,i])[0]]

        return np.sum(a[:,0]*a[:,1]) / np.sum(a[:,0])
             
        
    def apply_to_all_classes(self):
        nmCl = max(self.pred.shape[3] - 1,1)  
        return [directed_hausdorff(self.true[...,d], self.pred[...,d]) for d in range(nmCl)].mean()



def confusionMatrix(y_true,y_pred):
    
    yp1 = np.reshape(y_pred,[-1,1])
    yt1 = np.reshape(y_true,[-1,1])

    D = confusion_matrix(yt1, yp1 > 0.5)

    class metrics:
        def __init__(self, D):
                
            TN, FP, FN , TP = D[0,0] , D[0,1] , D[1,0] , D[1,1]
            self.Recall = TP / (TP + FN)
            self.Sensitivity = TP / (TP + FN)
            self.Specificity = TN/(TN + FP)
            self.Precision   = TP/(TP + FP)
    
    return metrics(D)

def ROC_Curve(y_true,y_pred):
    yp1 = np.reshape(y_pred,[-1,1])
    yt1 = np.reshape(y_true,[-1,1])

    fpr, tpr, _ = roc_curve(yt1, yp1,pos_label=[0])
    auc(fpr, tpr)
    # plt.plot(fpr, tpr)

def Precision_Recall_Curve(y_true=[],y_pred=[], Show=True, name='', directory=''):

    yt1 = np.reshape(y_true,[-1,1])
    yp1 = np.reshape(y_pred,[-1,1])

    precision, recall, thresholds = precision_recall_curve(yt1, yp1)
    average_precision = average_precision_score(yt1, yp1)

    if Show:
        fig = plt.figure()
        plt.step(recall, precision, color='b', alpha=0.9, where='post')
        # plt.fill_between(recall, precision, alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.title(f'{name} AP={average_precision:0.2f}')
        plt.show()
        
        fig.savefig(directory + name + '.png')
    return precision, recall

