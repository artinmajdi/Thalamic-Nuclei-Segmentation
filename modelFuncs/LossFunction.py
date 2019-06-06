import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
import modelFuncs.Metrics as Metrics
from keras import losses # optimizers, metrics
import keras.backend as Keras_Backend
import numpy as np


def LossInfo(loss_Index):
    switcher = {
        1: (losses.binary_crossentropy      , 'Loss_BCE'),
        2: (losses.categorical_crossentropy , 'Loss_CCE'),
        3: (My_BCE_Loss                     , 'My_BCE_Loss'),
        4: (My_LogDice_Loss                 , 'My_LogDice_Loss'),
        5: (My_Joint_Loss                   , 'My_Joint_Loss'),        
    }
    return switcher.get(loss_Index, 'WARNING: Invalid loss function index')

def My_BCE_Loss(weights):

    def func_loss(y_true,y_pred):
        NUM_CLASSES = y_pred.shape[3] - 1
        loss = [ weights[d]*losses.binary_crossentropy(y_true[...,d],y_pred[...,d]) for d in range(NUM_CLASSES)]

        return tf.divide(tf.reduce_sum(loss), tf.cast(NUM_CLASSES,tf.float32))

    return func_loss

def My_LogDice_Loss(weights):

    def func_loss(y_true,y_pred):
        
        NUM_CLASSES = y_pred.shape[3] - 1
        loss = [ - weights[d]*tf.log( Metrics.mDice(y_true,y_pred) )   for d in range(NUM_CLASSES)]
        return tf.divide(tf.reduce_sum(loss), tf.cast(NUM_CLASSES,tf.float32))

    return func_loss

def My_Joint_Loss(weights):

    def func_loss(y_true,y_pred):
        NUM_CLASSES = y_pred.shape[3] - 1
        loss = [ weights[d] * (  losses.binary_crossentropy(y_true[...,d],y_pred[...,d]) - tf.log(Metrics.mDice(y_true[...,d],y_pred[...,d]))  ) for d in range(NUM_CLASSES)]

        return tf.divide(tf.reduce_sum(loss), tf.cast(NUM_CLASSES,tf.float32))

    return func_loss    


# def Loss_Log_Dice(y_true,y_pred):
#     return -tf.log( Metrics.mDice(y_true,y_pred) )

# def Loss_CCE_And_LogDice(y_true,y_pred):
#     return losses.categorical_crossentropy(y_true,y_pred) + Loss_Log_Dice(y_true,y_pred)



# def weightedBinaryCrossEntropy(y_true, y_pred):
#     weight = y_true*1e6
#     bce = Keras_Backend.binary_crossentropy(y_true, y_pred)
#     return Keras_Backend.mean(bce*weight)

# def myCross_entropy(y_true,y_pred):
#     n_class = 2
#     y_true = tf.reshape(y_true, [-1, n_class])
#     y_pred = tf.reshape(y_pred, [-1, n_class])
#     return -tf.reduce_mean(y_true*tf.log(tf.clip_by_value(y_pred,1e-10,1.0)), name="cross_entropy")
