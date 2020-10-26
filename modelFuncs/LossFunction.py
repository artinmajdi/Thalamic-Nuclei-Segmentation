import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
from keras import losses


def LossInfo(loss_Index):
    switcher = {
        1: (losses.binary_crossentropy, 'LS_BCE'),
        2: (losses.categorical_crossentropy, 'LS_CCE'),
        3: (My_BCE_Loss, 'LS_MyBCE'),
        4: (My_LogDice_Loss, 'LS_MyLogDice'),
        5: (My_Joint_Loss, 'LS_MyJoint'),
        6: (My_Joint_Loss_GMean, 'LS_MyJoint_GMean'),
        7: (My_Dice_Loss, 'LS_MyDice'),

    }
    return switcher.get(loss_Index, 'WARNING: Invalid loss function index')


def func_Loss_Dice(x, y):
    return 1 - (tf.reduce_sum(input_tensor=tf.multiply(x, y)) + 1e-7) * 2 / (tf.reduce_sum(input_tensor=x) + tf.reduce_sum(input_tensor=y) + 1e-7)


def func_Loss_LogDice(x, y):
    return -tf.math.log((tf.reduce_sum(input_tensor=tf.multiply(x, y)) + 1e-7) * 2 / (tf.reduce_sum(input_tensor=x) + tf.reduce_sum(input_tensor=y) + 1e-7))


def func_Average(loss, NUM_CLASSES):
    return tf.divide(tf.reduce_sum(input_tensor=loss), tf.cast(NUM_CLASSES, tf.float32))


def My_BCE_Loss(W):
    def func_loss(y_true, y_pred):
        NUM_CLASSES = y_pred.shape[3]
        loss = [W[d] * losses.binary_crossentropy(y_true[..., d], y_pred[..., d]) for d in range(NUM_CLASSES)]
        return func_Average(loss, NUM_CLASSES)

    return func_loss


def My_BCE_wBackground_Loss(W):
    def func_loss(y_true, y_pred):
        NUM_CLASSES = y_pred.shape[3]
        loss = [W[d] * losses.binary_crossentropy(y_true[..., d], y_pred[..., d]) for d in range(NUM_CLASSES)]
        return func_Average(loss, NUM_CLASSES)

    return func_loss


def My_BCE_unweighted_Loss(W):
    def func_loss(y_true, y_pred):
        NUM_CLASSES = y_pred.shape[3]
        loss = [losses.binary_crossentropy(y_true[..., d], y_pred[..., d]) for d in range(NUM_CLASSES)]
        return func_Average(loss, NUM_CLASSES)

    return func_loss


def My_LogDice_Loss(W):
    def func_loss(y_true, y_pred):
        NUM_CLASSES = y_pred.shape[3]
        loss = [func_Loss_LogDice(y_true[..., d], y_pred[..., d]) for d in range(NUM_CLASSES)]
        return func_Average(loss, NUM_CLASSES)

    return func_loss


def My_Dice_Loss(W):
    def func_loss(y_true, y_pred):
        NUM_CLASSES = y_pred.shape[3]
        loss = [func_Loss_Dice(y_true[..., d], y_pred[..., d]) for d in range(NUM_CLASSES)]
        return func_Average(loss, NUM_CLASSES)

    return func_loss


def My_LogDice_wBackground_Loss(W):
    def func_loss(y_true, y_pred):
        NUM_CLASSES = y_pred.shape[3]
        loss = [W[d] * func_Loss_LogDice(y_true[..., d], y_pred[..., d]) for d in range(NUM_CLASSES)]
        return func_Average(loss, NUM_CLASSES)

    return func_loss


def My_LogDice_unweighted_Loss(W):
    def func_loss(y_true, y_pred):
        NUM_CLASSES = y_pred.shape[3]
        loss = [func_Loss_LogDice(y_true[..., d], y_pred[..., d]) for d in range(NUM_CLASSES)]
        return func_Average(loss, NUM_CLASSES)

    return func_loss


def My_LogDice_unweighted_WoBackground_Loss(W):
    def func_loss(y_true, y_pred):
        NUM_CLASSES = y_pred.shape[3] - 1
        loss = [func_Loss_LogDice(y_true[..., d], y_pred[..., d]) for d in range(NUM_CLASSES)]
        return func_Average(loss, NUM_CLASSES)

    return func_loss


def My_Joint_Loss(W):
    def func_loss(y_true, y_pred):
        NUM_CLASSES = y_pred.shape[3]
        loss = [W[d] * losses.binary_crossentropy(y_true[..., d], y_pred[..., d]) + func_Loss_LogDice(y_true[..., d],
                                                                                                      y_pred[..., d])
                for d in range(NUM_CLASSES)]

        return func_Average(loss, NUM_CLASSES)

    return func_loss


def My_Joint_Loss_GMean(W):  # geometrical mean

    def func_loss(y_true, y_pred):
        NUM_CLASSES = y_pred.shape[3]

        DiceLoss = [func_Loss_LogDice(y_true[..., d], y_pred[..., d]) for d in range(NUM_CLASSES)]
        wBCE = [W[d] * losses.binary_crossentropy(y_true[..., d], y_pred[..., d]) for d in range(NUM_CLASSES)]

        if func_Average(DiceLoss, NUM_CLASSES) is None:
            return func_Average(wBCE, NUM_CLASSES)
        else:
            loss = [DiceLoss[d] * wBCE[d] for d in range(NUM_CLASSES)]  # tf.multiply(DiceLoss,wBCE)
            return tf.sqrt(func_Average(loss, NUM_CLASSES))

    return func_loss
