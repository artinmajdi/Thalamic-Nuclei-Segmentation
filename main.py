import os, sys
sys.path.append(os.path.dirname(__file__))
from otherFuncs import smallFuncs
from tqdm import tqdm
from modelFuncs import choosingModel
from keras.models import load_model
from Parameters import paramFunc


# TODO:   make a new functions for reading all dices for each test cases and put them in a table for each nuclei
# TODO:    write a new function taht could raed the history files and plot the dice, loss for trainign and validation



#! this is for experimemnt cases
def readingTheParams():

    from Parameters import UserInfo
    UserInfo = smallFuncs.terminalEntries(UserInfo=UserInfo.__dict__)
    AllParamsList = loadExperiments(UserInfo)

    return AllParamsList

def subDict(UserInfoB, ExpList):
    for entry in list(ExpList.keys()):
        UserInfoB[entry] = ExpList[entry]
    return UserInfoB

def loadExperiments(UserInfo_Orig):

    AllParamsList = {}
    ExpList = UserInfo_Orig['AllExperimentsList']
    for Keys in list(ExpList.keys()):

        UserInfo = subDict(UserInfo_Orig, ExpList[Keys])

        A = paramFunc.Run(UserInfo)
        AllParamsList[Keys] = A

    return AllParamsList

def check_Dataset(params, flag, Info):  #  mode = 'experiments' # single_run'

    if flag:
        mode = 'experiment'
        from otherFuncs import datasets
        from preprocess import applyPreprocess

        #! copying the dataset into the experiment folder
        if params.preprocess.CreatingTheExperiment: datasets.movingFromDatasetToExperiments(params)


        #! preprocessing the data
        if params.preprocess.Mode:
            applyPreprocess.main(params, mode)
            params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)


        #! correcting the number of layers
        num_Layers = smallFuncs.correctNumLayers(params)
        params.WhichExperiment.HardParams.Model.num_Layers = num_Layers

        #! Finding the final image sizes after padding & amount of padding
        Subjects_Train, Subjects_Test, new_inputSize = smallFuncs.imageSizesAfterPadding(params, mode)

        params.directories.Train.Input.Subjects = Subjects_Train
        params.directories.Test.Input.Subjects  = Subjects_Test
        params.WhichExperiment.HardParams.Model.InputDimensions = new_inputSize


        # params.preprocess.TestOnly = True
        #! loading the dataset
        Data = datasets.loadDataset(params)
        params.WhichExperiment.HardParams.Model.imageInfo = Data.Info

        Info['num_Layers']     = num_Layers
        Info['new_inputSize']  = new_inputSize
        Info['Subjects_Train'] = Subjects_Train
        Info['Subjects_Test']  = Subjects_Test
        Info['imageInfo']      = Data.Info

        return Data, params, Info

    else:
        params.WhichExperiment.HardParams.Model.num_Layers      = Info['num_Layers']
        params.WhichExperiment.HardParams.Model.imageInfo       = Info['imageInfo']
        params.directories.Train.Input.Subjects                 = Info['Subjects_Train']
        params.directories.Test.Input.Subjects                  = Info['Subjects_Test']
        params.WhichExperiment.HardParams.Model.InputDimensions = Info['new_inputSize']

        return '', params, ''

def check_Run(params, Data):

    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index

    # params.preprocess.TestOnly = True
    if not params.preprocess.TestOnly:
        #! Training the model
        smallFuncs.Saving_UserInfo(params.directories.Train.Model, params, params.UserInfo)
        model = choosingModel.architecture(params)
        model, hist = choosingModel.modelTrain(Data, params, model)


        smallFuncs.saveReport(params.directories.Train.Model , 'hist_history' , hist.history , params.UserInfo['SaveReportMethod'])
        smallFuncs.saveReport(params.directories.Train.Model , 'hist_model'   , hist.model   , params.UserInfo['SaveReportMethod'])
        smallFuncs.saveReport(params.directories.Train.Model , 'hist_params'  , hist.params  , params.UserInfo['SaveReportMethod'])

    else:
        # TODO: I need to think more about this, why do i need to reload params even though i already have to load it in the beggining of the code
        #! loading the params
        params.UserInfo = smallFuncs.Loading_UserInfo(params.directories.Train.Model + '/UserInfo.mat', params.UserInfo['SaveReportMethod'])
        params = paramFunc.Run(params.UserInfo)
        params.WhichExperiment.HardParams.Model.InputDimensions = params.UserInfo['InputDimensions']
        params.WhichExperiment.HardParams.Model.num_Layers      = params.UserInfo['num_Layers']

        #! loading the model
        model = load_model(params.directories.Train.Model + '/model.h5')



    #! Testing
    pred, Dice, score = {}, {}, {}
    for name in tqdm(Data.Test):
        ResultDir = params.directories.Test.Result
        padding = params.directories.Test.Input.Subjects[name].Padding
        Dice[name], pred[name], score[name] = choosingModel.applyTestImageOnModel(model, Data.Test[name], params, name, padding, ResultDir)



    #! training predictions
    if params.WhichExperiment.HardParams.Model.Measure_Dice_on_Train_Data:
        ResultDir = smallFuncs.mkDir(params.directories.Test.Result + '/TrainData_Output')
        for name in tqdm(Data.Train_ForTest):
            padding = params.directories.Train.Input.Subjects[name].Padding
            Dice[name], pred[name], score[name] = choosingModel.applyTestImageOnModel(model, Data.Train_ForTest[name], params, name, padding, ResultDir)

    return pred

def check_show(Data, pred):
    #! showing the outputs
    for ind in [10]: # ,13,17]:
        name = list(Data.Test)[ind]   # Data.Train_ForTest
        # name = 'vimp2_2039_03182016'
        smallFuncs.imShow( Data.Test[name].Image[ind,:,:,0] ,  Data.Test[name].OrigMask[...,ind,0]  ,  pred[name][...,ind,0] )

def gpuSetting(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params.WhichExperiment.HardParams.Machine.GPU_Index
    import tensorflow as tf
    from keras import backend as K
    K.set_session(tf.Session(   config=tf.ConfigProto( allow_soft_placement=True , gpu_options=tf.GPUOptions(allow_growth=True) )   ))
    return K


#! we assume that number of layers , and other things that might effect the input data stays constant
AllParamsList = readingTheParams()

#! reading the dataset
ind, params = list(AllParamsList.items())[0]
Data, params, Info = check_Dataset(params=params, flag=True, Info={})

for ind, params in list(AllParamsList.items()):

    K = gpuSetting(params)

    _, params, _ = check_Dataset(params=params, flag=False, Info=Info)

    pred = check_Run(params, Data)

    if 0: check_show(Data, pred)

K.clear_session()