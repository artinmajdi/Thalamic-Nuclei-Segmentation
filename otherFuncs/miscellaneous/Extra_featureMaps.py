import numpy as np
import nibabel as nib
import os, sys
import keras.models as kerasmodels
import keras
import matplotlib.pyplot as plt
import skimage
import PIL
from tqdm import tqdm
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from Parameters import UserInfo
from Parameters import paramFunc
from otherFuncs import smallFuncs
from modelFuncs import choosingModel
from otherFuncs import datasets


UserInfo = UserInfo.__dict__  

UserInfo['CrossVal'].index   = ['a']
UserInfo['simulation'].num_Layers = 3
UserInfo['architectureType'] = 'Res_Unet2'
UserInfo['Experiments'].Index = '6'
UserInfo['copy_Thalamus'] = False
UserInfo['TypeExperiment'] = 15
UserInfo['simulation'].LR_Scheduler = True    
UserInfo['simulation'].FirstLayer_FeatureMap_Num = 20
UserInfo['simulation'].slicingDim = [2]
UserInfo['simulation'].nucleus_Index = [1,2,4,5,6,7,8,9,10,11,12,13,14]   
    
class LoadingData:
    def __init__(self): pass
        
    def PreMode(self, UserInfo, *args):  
        # self.UserInfoB = UserInfo.__dict__  
        self.UserInfoB = UserInfo 

        def UserInputs(self,args):  
            for ag in args: 
                if 'GPU' in ag[0]: self.UserInfoB['simulation'].GPU_Index = ag[1]
        UserInputs(self,args)

        def gpuSetting(self):            
            os.environ["CUDA_VISIBLE_DEVICES"] = self.params.WhichExperiment.HardParams.Machine.GPU_Index
            import tensorflow as tf
            from keras import backend as K
            K.set_session(tf.compat.v1.Session(   config=tf.compat.v1.ConfigProto( allow_soft_placement=True , gpu_options=tf.compat.v1.GPUOptions(allow_growth=True) )   ))
            self.K = K
            
        # self.UserInfoB = smallFuncs.terminalEntries(self.UserInfoB)
        self.UserInfoB['simulation'].TestOnly = True
        self.params = paramFunc.Run(self.UserInfoB, terminal=True)

        return gpuSetting(self)
        
    def ReadData(self):
        self.params = paramFunc.Run(self.UserInfoB, terminal=False)
        self.Data, self.params = datasets.loadDataset(self.params)
        self.params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data = False
        return self.Data

    def LoadModel(self):
        self.params = paramFunc.Run(self.UserInfoB, terminal=False)
        self.model = kerasmodels.load_model(self.params.directories.Train.Model + '/model.h5') # + '/model_CSFn.h5')

        class Layers:
            Outputs = [layer.output for layer in self.model.layers[1:]]
            Names   = [layer.name for layer in self.model.layers[1:]]
            Input   = self.model.layers[0].output # keras.layers.Input( tuple(self.params.WhichExperiment.HardParams.Model.InputDimensions[:self.params.WhichExperiment.HardParams.Model.Method.InputImage2Dvs3D]) + (1,) )
            predictions = ''
        self.Layers = Layers()

        self.model_wAllLayers = keras.models.Model(inputs=self.Layers.Input, outputs=self.Layers.Outputs)

class Visualize_FeatureMaps(LoadingData):

    def predict(self , subject_Index , slice):                                
        class subject_cls:
            Subject_Index = subject_Index
            Slice = slice
            name = list(self.Data.Test)[subject_Index]
            Image = self.Data.Test[name].Image[None,slice,...]
            Mask  = self.Data.Test[name].Mask[None,slice,...]
                                      
        self.subject = subject_cls()
        self.Layers.predictions = self.model_wAllLayers.predict( self.subject.Image )
        return self.Layers.predictions
    
    def concatenate_pred(self,layer_num):

        class oneLayerInfo:
            def __init__(self, layer_name , Image):
                self.layer_num  = layer_num
                self.layer_name = layer_name 
                self.Image      = Image
                       
        def concatenate_main(self):
            class concatenate_Info:
                pred2        = self.Layers.predictions[layer_num][0,...]
                SzX, SzY, num_features = pred2.shape[:3]
                Num_Columns  = int(np.sqrt(num_features))
                Num_Rows     = int(np.ceil(num_features/Num_Columns))                            
            CI = concatenate_Info()
            
            FullImage = np.zeros((CI.SzX* CI.Num_Rows, CI.SzY*CI.Num_Columns))

            for i in range(CI.Num_Rows):

                vertical_Range = range( i*CI.Num_Columns , min((i+1)*CI.Num_Columns,CI.num_features) )
                horizont_Range = range( CI.SzX* i , CI.SzX*(i+1) )

                oneRow = np.concatenate(CI.pred2.transpose([2,0,1])[vertical_Range , ...] , axis=1)
                FullImage[horizont_Range ,:oneRow.shape[1]] = oneRow

            return FullImage
        FullImage = concatenate_main(self)

        self.oneLayerInfo = oneLayerInfo( self.Layers.Names[layer_num] , FullImage)

    def show(self,*args):
        cmap = args[0] if len(args) != 0 else 'gray'

        plt.figure()
        plt.imshow(self.oneLayerInfo.Image,cmap=cmap)  
        plt.title(self.oneLayerInfo.layer_name)
        plt.show()  

    def save_OneLayer(self):
        def normalize(im):
            mn , mx = im.min() , im.max()
            image = 256*(im - mn)/(mx-mn)
            return image.astype(np.uint8)

        image = normalize(self.oneLayerInfo.Image)
        # print(self.subject.name, self.subject.Slice , 'saving layer:',self.oneLayerInfo.layer_name)
        smallFuncs.mkDir(self.params.directories.Train.Model + '/FeatureMaps/' + self.subject.name + '/' + str(self.subject.Slice) )
        im = PIL.Image.fromarray(image).save(self.params.directories.Train.Model + '/FeatureMaps/' + self.subject.name + '/' + str(self.subject.Slice) + '/' + str(self.oneLayerInfo.layer_num) + '_' + self.oneLayerInfo.layer_name + '.png')
        
    def save_All_Layers(self):
        for layer_num in tqdm(range(len(self.Layers.Names)),desc='saving All Layers ' + self.subject.name + ' Slice ' + str(self.subject.Slice)):
            self.concatenate_pred(layer_num)
            self.save_OneLayer()
       
VFM = Visualize_FeatureMaps()
VFM.PreMode(UserInfo,('GPU',"5"))  
Nuclei_Indexes = VFM.UserInfoB['simulation'].nucleus_Index.copy()

print('ADDRESS',VFM.params.directories.Train.Model)
for x in [1]: # Nuclei_Indexes:  
    VFM.UserInfoB['simulation'].nucleus_Index = [x]
    print('nucleus' , VFM.UserInfoB['simulation'].nucleus_Index )    
    VFM.ReadData()
    VFM.LoadModel()

    for subject_Index in range(len(list(VFM.Data.Test))):
        a = VFM.Data.Test[list(VFM.Data.Test)[subject_Index]].Image.shape[0]
        for slice in [int(a/2)]:
            VFM.predict(subject_Index=subject_Index , slice=slice)
            # VFM.concatenate_pred(layer_num=6)
            # VFM.show('gray')
            VFM.save_All_Layers()



print('---')

os.system('bash /bashCodes/zip_Bash_FeatMps')

VFM.K.clear_session()

