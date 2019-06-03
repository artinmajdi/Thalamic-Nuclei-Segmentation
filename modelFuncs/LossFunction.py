import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
import modelFuncs.Metrics as Metrics
from keras import losses # optimizers, metrics
import keras.backend as Keras_Backend
import numpy as np

def LossInfo(loss_Index):
    switcher = {
        1: (losses.binary_crossentropy, 'Loss_BCE'),
        2: (Loss_Dice, 'Loss_Dice'),
        3: (Loss_Log_Dice, 'Loss_LogDice'),
        4: (Loss_CCE_And_LogDice, 'Loss_CCE_And_LogDice'),
        5: (losses.categorical_crossentropy, 'Loss_CCE'),
        
    }
    return switcher.get(loss_Index, 'WARNING: Invalid loss function index')

# TODO: check to see if having this import inside this function slows down the code
def MyLoss_binary_crossentropy(y_true,y_pred):

    loss = 0
    nmCl = y_pred.shape[3] - 1
    for d in range(nmCl):
        loss = loss + losses.binary_crossentropy(y_true[...,d],y_pred[...,d])

    return tf.divide(loss,tf.cast(nmCl,tf.float32))

def Loss_Dice(y_true,y_pred):
    return 1 - Metrics.mDice(y_true,y_pred)

def Loss_Log_Dice(y_true,y_pred):
    return tf.log( 1 - Metrics.mDice(y_true,y_pred) )

def Loss_CCE_And_LogDice(y_true,y_pred):
    return losses.categorical_crossentropy(y_true,y_pred) + Loss_Log_Dice(y_true,y_pred)



# def weightedBinaryCrossEntropy(y_true, y_pred):
#     weight = y_true*1e6
#     bce = Keras_Backend.binary_crossentropy(y_true, y_pred)
#     return Keras_Backend.mean(bce*weight)

# def myCross_entropy(y_true,y_pred):
#     n_class = 2
#     y_true = tf.reshape(y_true, [-1, n_class])
#     y_pred = tf.reshape(y_pred, [-1, n_class])
#     return -tf.reduce_mean(y_true*tf.log(tf.clip_by_value(y_pred,1e-10,1.0)), name="cross_entropy")
