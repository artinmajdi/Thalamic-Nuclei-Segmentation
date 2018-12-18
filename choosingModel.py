from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Reshape, Flatten, BatchNormalization, Input, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
from otherFuncs.smallFuncs import mkDir


# ! main Function
def modelTrain(Data, params, model):
    ModelParam = params.directories.Experiment.HardParams.Model
    model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss, metrics=ModelParam.metrics)

    # if the shuffle argument in model.fit is set to True (which is the default), the training data will be randomly shuffled at each epoch.
    if ModelParam.Validation.fromKeras:
        hist = model.fit(x=Data.Train.Image, y=Data.Train.Label, batch_size=ModelParam.batch_size, epochs=ModelParam.epochs, shuffle=True, validation_split=ModelParam.Validation.Percentage)
    else:
        hist = model.fit(x=Data.Train.Image, y=Data.Train.Label, batch_size=ModelParam.batch_size, epochs=ModelParam.epochs, shuffle=True, validation_data=(Data.Validation.Image, Data.Validation.Label))

    mkDir(params.directories.Train.Model)
    model.save(params.directories.Train.Model + '/model.h5', overwrite=True, include_optimizer=True )

    if ModelParam.showHistory: print(hist.history)

    return model


def architecture(Data, params):
    params.directories.Experiment.HardParams.Model.imageInfo = Data.Info
    ModelParam = params.directories.Experiment.HardParams.Model
    if 'U-Net' in ModelParam.architectureType:
        model = UNet(ModelParam)

    elif 'MLP' in ModelParam.architectureType:
        ModelParam.numClasses = Data.Train.Label.shape[1] # len(np.unique(Train.Label))
        model = CNN(ModelParam)

    model.summary()
    return model, params


def Unet_sublayer_Contracting(inputs, nL, Modelparam):
    conv = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(inputs)
    conv = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)
    pool = MaxPooling2D( pool_size=Modelparam.MaxPooling.pool_size)(conv)
    if Modelparam.Dropout.Mode: pool = Dropout(Modelparam.Dropout.Value)(pool)
    return pool, conv

def Unet_sublayer_Expanding(inputs, nL, Modelparam, contractingInfo):
    UP = Conv2DTranspose(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.convTranspose, strides=(2,2), padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(inputs)
    UP = concatenate([UP,contractingInfo[nL+1]],axis=3)
    conv = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(UP)
    conv = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv)
    if Modelparam.Dropout.Mode: conv = Dropout(Modelparam.Dropout.Value)(conv)
    return conv


# ! U-Net Architecture
def UNet(Modelparam):
    inputs = Input( (Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1) )
    WeightBiases = inputs
    # TODO do I need to have BatchNormalization in each layer?
    if Modelparam.batchNormalization:  WeightBiases = BatchNormalization()(WeightBiases)

    # ! contracting layer
    ConvOutputs = {}
    for nL in range(Modelparam.num_Layers -1):
        WeightBiases, ConvOutputs[nL+1] = Unet_sublayer_Contracting(WeightBiases, nL, Modelparam)

    # ! middle layer
    nL = Modelparam.num_Layers - 1
    WeightBiases = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(WeightBiases)
    WeightBiases = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(WeightBiases)
    if Modelparam.Dropout.Mode: WeightBiases = Dropout(Modelparam.Dropout.Value)(WeightBiases)

    # ! expanding layer
    for nL in reversed(range(Modelparam.num_Layers -1)):
        WeightBiases = Unet_sublayer_Expanding(WeightBiases, nL, Modelparam, ConvOutputs)

    # ! final outputing the data
    final = Conv2D(2, kernel_size=Modelparam.ConvLayer.Kernel_size.output, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.output)(WeightBiases)
    model = Model(inputs=[inputs], outputs=[final])

    return model

# U-Net without for loop
    # # ! layer 1
    # nL = 0,  num_Channels = 64*(2**nL)
    # conv1 = Conv2D(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.down, padding=Modelparam.ConvLayer.padding, activation=Modelparam.ConvLayer.activitation)(WeightBiases)
    # conv1 = Conv2D(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.down, padding=Modelparam.ConvLayer.padding, activation=Modelparam.ConvLayer.activitation)(conv1)
    # pool1 = MaxPooling2D( pool_size=Modelparam.MaxPooling.pool_size)(conv1)
    # if Modelparam.Dropout.Mode: pool1 = Dropout(Modelparam.Dropout.Value)(pool1)
    #
    # # ! layer 2
    # nL = nL + 1,  num_Channels = 64*(2**nL)
    # conv2 = Conv2D(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.down, padding=Modelparam.ConvLayer.padding, activation=Modelparam.ConvLayer.activitation)(pool1)
    # conv2 = Conv2D(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.down, padding=Modelparam.ConvLayer.padding, activation=Modelparam.ConvLayer.activitation)(conv2)
    # pool2 = MaxPooling2D( pool_size=Modelparam.MaxPooling.pool_size)(conv2)
    # if Modelparam.Dropout.Mode: pool2 = Dropout(Modelparam.Dropout.Value)(pool2)
    #
    # # ! layer 3
    # nL = nL + 1,  num_Channels = 64*(2**nL)
    # conv3 = Conv2D(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.down, padding=Modelparam.ConvLayer.padding, activation=Modelparam.ConvLayer.activitation)(pool2)
    # conv3 = Conv2D(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.down, padding=Modelparam.ConvLayer.padding, activation=Modelparam.ConvLayer.activitation)(conv3)
    # pool3 = MaxPooling2D( pool_size=Modelparam.MaxPooling.pool_size)(conv3)
    # if Modelparam.Dropout.Mode: pool3 = Dropout(Modelparam.Dropout.Value)(pool3)
    #
    # # ! layer 4
    # nL = nL + 1,  num_Channels = 64*(2**nL)
    # conv4 = Conv2D(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.down, padding=Modelparam.ConvLayer.padding, activation=Modelparam.ConvLayer.activitation)(pool3)
    # conv4 = Conv2D(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.down, padding=Modelparam.ConvLayer.padding, activation=Modelparam.ConvLayer.activitation)(conv4)
    # pool4 = MaxPooling2D( pool_size=Modelparam.MaxPooling.pool_size)(conv4)
    # if Modelparam.Dropout.Mode: pool4 = Dropout(Modelparam.Dropout.Value)(pool4)
    #
    # # ! middle layer
    # nL = nL + 1,  num_Channels = 64*(2**nL)
    # conv5 = Conv2D(64*(2**(Modelparam.num_Layers - 1)), kernel_size=Modelparam.ConvLayer.Kernel_size.down, strides=Modelparam.ConvLayer.strides, padding=Modelparam.ConvLayer.padding, activation=Modelparam.ConvLayer.activitation)(pool4)
    # conv5 = Conv2D(64*(2**(Modelparam.num_Layers - 1)), kernel_size=Modelparam.ConvLayer.Kernel_size.down, strides=Modelparam.ConvLayer.strides, padding=Modelparam.ConvLayer.padding, activation=Modelparam.ConvLayer.activitation)(conv5)
    # if Modelparam.Dropout.Mode: conv5 = Dropout(Modelparam.Dropout.Value)(conv5)
    #
    # # ! expanding layer 4 nL = 3
    # nL = nL - 1,  num_Channels = 64*(2**nL)
    # up4 = Conv2DTranspose(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.convTranspose, strides=(2,2), padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(WeightBiases)
    # up4 = concatenate([up4,ConvOutputs[nL+1]],axis=3)
    # conv4u = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(up4)
    # conv4u = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv4u)
    #
    # # ! expanding layer 3 nL = 2
    # nL = nL - 1,  num_Channels = 64*(2**nL)
    # up3 = Conv2DTranspose(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.convTranspose, strides=(2,2), padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv4u)
    # up3 = concatenate([up4,ConvOutputs[nL+1]],axis=3)
    # conv3u = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(up3)
    # conv3u = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv3u)
    #
    # # ! expanding layer 2 nL = 1
    # nL = nL - 1,  num_Channels = 64*(2**nL)
    # up2 = Conv2DTranspose(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.convTranspose, strides=(2,2), padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv3u)
    # up2 = concatenate([up4,ConvOutputs[nL+1]],axis=3)
    # conv2u = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(up2)
    # conv2u = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv2u)
    #
    # # ! expanding layer 1 nL = 0
    # nL = nL - 1,  num_Channels = 64*(2**nL)
    # up1 = Conv2DTranspose(num_Channels, kernel_size=Modelparam.ConvLayer.Kernel_size.convTranspose, strides=(2,2), padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv2u)
    # up1 = concatenate([up4,ConvOutputs[nL+1]],axis=3)
    # conv1u = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(up1)
    # conv1u = Conv2D(64*(2**nL), kernel_size=Modelparam.ConvLayer.Kernel_size.conv, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.layers)(conv1u)
    #
    # # ! last part; outputing the data
    # final = Conv2D(2, kernel_size=Modelparam.ConvLayer.Kernel_size.output, padding=Modelparam.ConvLayer.padding, activation=Modelparam.Activitation.output)(conv1u)

# def get_unet():
    # height,width = 512, 512
    # inputs = Input((height,width, 1))
    # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    # conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    # conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    # conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    # conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    # conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    # up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    # conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    # up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    # conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    # up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    # conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    # model = Model(inputs=[inputs], outputs=[conv10])
    # model.compile(optimizer=ModelParam.optimizer, loss=ModelParam.loss, metrics=ModelParam.metrics)

    # return model


# ! CNN Architecture
def CNN(Modelparam):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=Modelparam.kernel_size, padding=Modelparam.padding, activation=Modelparam.activitation, input_shape=(Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1)))
    model.add(MaxPooling2D(pool_size=Modelparam.MaxPooling.pool_size))
    model.add(Dropout(Modelparam.Dropout.Value))

    model.add(Conv2D(filters=8, kernel_size=Modelparam.kernel_size, padding=Modelparam.padding, activation=Modelparam.activitation, input_shape=(Modelparam.imageInfo.Height, Modelparam.imageInfo.Width, 1)))
    model.add(MaxPooling2D(pool_size=Modelparam.MaxPooling.pool_size))
    model.add(Dropout(Modelparam.Dropout.Value))

    model.add(Flatten())
    model.add(Dense(128 , activation=Modelparam.Activitation.layers))
    model.add(Dropout(Modelparam.Dropout.Value))
    model.add(Dense(Modelparam.numClasses , activation=Modelparam.Activitation.output))

    return model
