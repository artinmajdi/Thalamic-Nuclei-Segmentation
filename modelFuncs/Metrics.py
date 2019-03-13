
import tensorflow as tf

def MetricInfo(Metric_Index):
    switcher = {
        1: (Dice_Calculator, 'Dice'),
        2: ('acc', 'Acc'),
        3: (['acc',Dice_Calculator], 'Acc_N_Dice')
    }
    return switcher.get(Metric_Index, 'WARNING: Invalid metric index')


def Dice_Calculator(y_true,y_pred):

    Dice = 0
    nmCl = y_pred.shape[3] - 1
    for d in range(nmCl):
        Dice = Dice + tf.reduce_sum(tf.multiply(y_true[...,d],y_pred[...,d]))*2/( tf.reduce_sum(y_true[...,d]) + tf.reduce_sum(y_pred[...,d]) + 1e-5)
    # Dice = Dice / tf.cast(nmCl,tf.float32)
    return tf.divide(Dice,tf.cast(nmCl,tf.float32))


# def Dice_Calculator_works(y_true,y_pred):
#     d = y_pred.shape[3] - 1
#     y_true2 = y_true[...,:d]
#     y_pred2 = y_pred[...,:d]
#     Dice = tf.reduce_sum(tf.multiply(y_true2,y_pred))*2/( tf.reduce_sum(y_true2) + tf.reduce_sum(y_pred) + 1e-5)
#     #! WORKED without below
#     # if Dice is None:
#     #     Dice = tf.constant(0,dtype=tf.float32)
#     return Dice