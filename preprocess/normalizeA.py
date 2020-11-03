import numpy as np

def funcNormalize(Method , Image):

    if Method == 'MinMax':
        Image = np.float32(Image)
        Image = ( Image-Image.min() )/( Image.max() - Image.min() )

    elif Method == '1Std0Mean':
        Image = np.float32(Image)
        Image = ( Image-Image.mean() )/( Image.std() )

    elif Method == 'Both':
        Image = np.float32(Image)
        Image = ( Image-Image.min() )/( Image.max() - Image.min() )
        Image = ( Image-Image.mean() )/( Image.std() )

    return Image

def main_normalize(Normalize , Image):

    if Normalize.Mode:
        Image = funcNormalize(Normalize.Method , Image)

    return Image


