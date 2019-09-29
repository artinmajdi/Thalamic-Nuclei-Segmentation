# import nilearn
import nibabel
import keras
import tensorflow as tf

# dir = '/array/ssd/msmajdi/experiments/keras/exp6/models/Old_Results_July9/Extra_Experiments/Extra_Experiments_Ver4_July7/ResUnet2_Dice_Loss/Main/ResUnet2_wLRScheduler/sE12_Cascade_FM20_Res_Unet2_NL3_LS_MyLogDice_US1_wLRScheduler_Main_Init_3T_CV_a/MultiClass_24567891011121314/sd2/'
# model = keras.models.load_model(dir + 'model.h5')
# model.summary()


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


print('---')