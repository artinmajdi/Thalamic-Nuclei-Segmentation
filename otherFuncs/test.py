# import nilearn
import nibabel as nib
import keras
import tensorflow as tf
from skimage.transform import AffineTransform , warp


# dir = '/array/ssd/msmajdi/experiments/keras/exp6/models/Old_Results_July9/Extra_Experiments/Extra_Experiments_Ver4_July7/ResUnet2_Dice_Loss/Main/ResUnet2_wLRScheduler/sE12_Cascade_FM20_Res_Unet2_NL3_LS_MyLogDice_US1_wLRScheduler_Main_Init_3T_CV_a/MultiClass_24567891011121314/sd2/'
# model = keras.models.load_model(dir + 'model.h5')
# model.summary()


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


def upsample_Image(Image, Mask , scale):
    szI = Image.shape
    szM = Mask.shape
    
    Image3 = np.zeros( (szI[0] , scale*szI[1] , scale*szI[2] , szI[3])  )
    Mask3  = np.zeros( (szM[0] , scale*szM[1] , scale*szM[2] , szM[3])  )

    newShape = (scale*szI[1] , scale*szI[2])

    # for i in range(Image.shape[2]):
    #     Image2[...,i] = scipy.misc.imresize(Image[...,i] ,size=newShape[:2] , interp='cubic')
    #     Mask2[...,i]  = scipy.misc.imresize( (Mask[...,i] > 0.5).astype(np.float32) ,size=newShape[:2] , interp='bilinear')

    tform = AffineTransform(scale=(scale, scale))
    for i in range(Image.shape[0]):

        for ch in range(Image3.shape[3]):
            Image3[i ,: ,: ,ch] = warp( np.squeeze(Image[i ,: ,: ,ch]), tform.inverse, output_shape=newShape, order=3)

        for ch in range(Mask3.shape[3]):
            Mask3[i ,: ,: ,ch]  = warp( (np.squeeze(Mask[i ,: ,: ,ch]) > 0.5).astype(np.float32) ,  tform.inverse, output_shape=newShape, order=0)
    
    return Image3 , Mask3