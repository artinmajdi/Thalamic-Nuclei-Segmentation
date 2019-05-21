import numpy as np
import nibabel as nib
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
import Parameters.UserInfo as UserInfo
import Parameters.paramFunc as paramFunc
import otherFuncs.smallFuncs as smallFuncs
import modelFuncs.choosingModel as choosingModel
import otherFuncs.datasets as datasets
import keras.models as kerasmodels
import keras
import matplotlib.pyplot as plt
import skimage
import PIL
from tqdm import tqdm

class LoadingData:
    def __init__(self): pass
        
    def PreMode(self, UserInfo, *args):  
        self.UserInfoB = UserInfo.__dict__  

        def UserInputs(self,args):  
            for ag in args: 
                if 'GPU' in ag[0]: self.UserInfoB['simulation'].GPU_Index = ag[1]
        UserInputs(self,args)

        def gpuSetting(self):            
            os.environ["CUDA_VISIBLE_DEVICES"] = self.params.WhichExperiment.HardParams.Machine.GPU_Index
            import tensorflow as tf
            from keras import backend as K
            K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))
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

        # model = choosingModel.architecture(params)
        # model.load_weights(dir + '1-THALAMUS/model_weights.h5')
        self.model = kerasmodels.load_model(self.params.directories.Train.Model + '/model_CSFn.h5')

        class Layers:
            Outputs = [layer.output for layer in self.model.layers[1:]]
            Names      = [layer.name for layer in self.model.layers[1:]]
            Input      = self.model.layers[0].output # keras.layers.Input( tuple(self.params.WhichExperiment.HardParams.Model.InputDimensions[:self.params.WhichExperiment.HardParams.Model.Method.InputImage2Dvs3D]) + (1,) )
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
for VFM.UserInfoB['simulation'].nucleus_Index in Nuclei_Indexes:  
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

os.system('bash /array/ssd/msmajdi/code/thalamus/keras/bashCodes/zip_Bash_FeatMps')

# keras.utils.plot_model(model,to_file=params.directories.Train.Model+'/Architecture.png',show_layer_names=True,show_shapes=True)
VFM.K.clear_session()

