import os, sys
# __file__ = '/array/ssd/msmajdi/code/thalamus/keras/'  #! only if I'm using Hydrogen Atom
sys.path.append(os.path.dirname(__file__))
from otherFuncs import smallFuncs





def __init__():

    from Parameters import UserInfo, paramFunc
    UserInfo = smallFuncs.terminalEntries(UserInfo=UserInfo)
    params = paramFunc.__init__(UserInfo)
    return params



def check_Dataset(params, mode):

    from otherFuncs import datasets
    from preprocess import applyPreprocess

    #! copying the dataset into the experiment folder
    if params.preprocess.CreatingTheExperiment: datasets.movingFromDatasetToExperiments(params)


    #! preprocessing the data
    if params.preprocess.Mode:
        applyPreprocess.main(params, mode)
        params.directories = smallFuncs.funcExpDirectories(params.WhichExperiment)


    #! correcting the number of layers
    params = smallFuncs.correctNumLayers(params)


    #! Finding the final image sizes after padding & amount of padding
    params = smallFuncs.imageSizesAfterPadding(params, mode)


    # params.preprocess.TestOnly = True
    #! loading the dataset
    Data, params = datasets.loadDataset(params)

    return Data, params



def check_Run(params, Data):

    from tqdm import tqdm
    from modelFuncs import choosingModel
    from keras.models import load_model

    #! configing the GPU
    K = smallFuncs.gpuConfig(params.WhichExperiment.HardParams.Machine.GPU_Index)

    # params.preprocess.TestOnly = True
    if not params.preprocess.TestOnly:
        #! Training the model
        smallFuncs.Saving_UserInfo(params.directories.Train.Model, params, params.UserInfo)
        model = choosingModel.architecture(params)
        model, hist = choosingModel.modelTrain(Data, params, model)


        smallFuncs.saveReport(params.directories.Train.Model , 'hist_history' , hist.history , params.UserInfo.SaveReportMethod)
        smallFuncs.saveReport(params.directories.Train.Model , 'hist_model'   , hist.model   , params.UserInfo.SaveReportMethod)
        smallFuncs.saveReport(params.directories.Train.Model , 'hist_params'  , hist.params  , params.UserInfo.SaveReportMethod)

    else:
        # TODO: I need to think more about this, why do i need to reload params even though i already have to load it in the beggining of the code
        #! loading the params
        params.UserInfo = smallFuncs.Loading_UserInfo(params.directories.Train.Model + '/UserInfo.mat', params.UserInfo.SaveReportMethod)
        params = paramFunc.__init__(params.UserInfo)
        params.WhichExperiment.HardParams.Model.InputDimensions = params.UserInfo.InputDimensions
        params.WhichExperiment.HardParams.Model.num_Layers      = params.UserInfo.num_Layers

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

    K.clear_session()

    return pred



def check_show(Data, pred):
    #! showing the outputs
    for ind in [10]: # ,13,17]:
        name = list(Data.Test)[ind]   # Data.Train_ForTest
        # name = 'vimp2_2039_03182016'
        smallFuncs.imShow( Data.Test[name].Image[ind,:,:,0] ,  Data.Test[name].OrigMask[...,ind,0]  ,  pred[name][...,ind,0] )