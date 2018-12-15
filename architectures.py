from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Reshape, Flatten

def sublayer_Unet(model, num_Channels, Modelparam):
    model.add(Conv2D(num_Channels, kernel_size=(3, 3), strides=(1, 1), padding=Modelparam.padding, activation=Modelparam.Activitation))
    model.add(Conv2D(num_Channels, kernel_size=(3, 3), strides=(1, 1), padding=Modelparam.padding, activation=Modelparam.Activitation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    return model

def UNet(Modelparam):
    model = Sequential()
    for nL in range(Modelparam.Num_Layers -1):
        model = sublayer_Unet(model, 64*(2^nL) , Modelparam)

    model.add(Conv2D(64*(2^(Modelparam.Num_Layers - 1)), kernel_size=Modelparam.kernel_size,strides=(1, 1), padding=Modelparam.padding, activation=Modelparam.Activitation))
    model.add(Conv2D(64*(2^(Modelparam.Num_Layers - 1)), kernel_size=Modelparam.kernel_size,strides=(1, 1), padding=Modelparam.padding, activation=Modelparam.Activitation))
    return model



def CNN(Modelparam):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=Modelparam.kernel_size, padding=Modelparam.padding, activation=Modelparam.activitation, input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(Modelparam.dropout))

    model.add(Conv2D(filters=32, kernel_size=Modelparam.kernel_size, padding=Modelparam.padding, activation=Modelparam.activitation, input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(Modelparam.dropout))


    model.add(Flatten())
    model.add(Dense(256 , activation=Modelparam.activitation))
    model.add(Dropout(Modelparam.dropout))
    model.add(Dense(Modelparam.numClasses , activation='softmax'))

    model.summary()

    return model
