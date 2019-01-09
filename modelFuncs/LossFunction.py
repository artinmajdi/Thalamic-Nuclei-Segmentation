import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
from keras import losses # optimizers, metrics
from modelFuncs import Metrics



def LossInfo(loss_Index):
    switcher = {
        1: (Loss_Dice, 'Loss_Dice'),
        2: (Loss_binary_crossentropy, 'Loss_BCE'),
        3: (Loss_binary_And_Dice, 'Loss_Dice_N_BCE')
    }
    return switcher.get(loss_Index, 'WARNING: Invalid loss function index')



def Loss_binary_crossentropy(y_true,y_pred):
    loss = 0
    nmCl = tf.constant(y_pred.shape[3] - 1)
    for d in range(nmCl):
        loss = loss + losses.binary_crossentropy(y_true[...,d],y_pred[...,d])

    return tf.divide(loss,tf.cast(nmCl,tf.float32))



def Loss_Dice(y_true,y_pred):
    return 1 - Metrics.Dice_Calculator(y_true,y_pred)



def Loss_binary_And_Dice(y_true,y_pred):
    return Loss_binary_crossentropy(y_true,y_pred) + Loss_Dice(y_true,y_pred)




# def myCross_entropy(y_true,y_pred):
#     n_class = 2
#     y_true = tf.reshape(y_true, [-1, n_class])
#     y_pred = tf.reshape(y_pred, [-1, n_class])
#     return -tf.reduce_mean(y_true*tf.log(tf.clip_by_value(y_pred,1e-10,1.0)), name="cross_entropy")


