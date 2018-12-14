from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Dense2D

def sublayer_Unet(model , num_Channels , Modelparam):
    model.add(Conv2D(num_Channels,kernel_size=(3,3),strides=(1,1),padding=Modelparam.padding,activation=Modelparam.Activitation))
    model.add(Conv2D(num_Channels,kernel_size=(3,3),strides=(1,1),padding=Modelparam.padding,activation=Modelparam.Activitation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    return model

def UNet(Modelparam):
    model = Sequential()
    for nL in range(Modelparam.Num_Layers -1):
        model = sublayer_Unet(model , 64*(2^nL) , Modelparam)

    model.add(Conv2D( 64*(2^(Modelparam.Num_Layers-1)) ,kernel_size=Modelparam.kernel_size,strides=(1,1),padding=Modelparam.padding,activation=Modelparam.Activitation))
    model.add(Conv2D( 64*(2^(Modelparam.Num_Layers-1)) ,kernel_size=Modelparam.kernel_size,strides=(1,1),padding=Modelparam.padding,activation=Modelparam.Activitation))
    return model



def MLP(Modelparam):
    model = Sequential()
    model.add(Dense(32, activation=Modelparam.Activitation, input_shape=(28,28)))
    model.add(Dense(16 , activation=Modelparam.Activitation))
    model.add(Dense(16 , activation=Modelparam.Activitation))
    model.add(Dense(Modelparam.NumClasses , activation='sigmoid'))
    return model
