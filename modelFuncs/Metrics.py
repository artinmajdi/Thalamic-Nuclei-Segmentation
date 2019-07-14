import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf

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
    
    ms1 = np.reshape(msk1,[-1,1])
    ms2 = np.reshape(msk2,[-1,1])

    D = confusion_matrix(ms1, ms2)

