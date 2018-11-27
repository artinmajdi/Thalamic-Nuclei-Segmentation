import numpy as np

def funcNormalize(params , Image):

    if params.normalize.Method == 'MinMax':
        Image = np.float32(Image)
        Image = ( Image-Image.min() )/( Image.max() - Image.min() )

    elif params.normalize.Method == '1Std0Mean':
        Image = np.float32(Image)
        Image = ( Image-Image.mean() )/( Image.std() )    
            
    return Image

def normalizeMain(params , Image):

    if params.normalize:
        Image = funcNormalize(params , Image)

    return Image


