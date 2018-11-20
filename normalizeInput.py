import numpy as np

def funcNormalize(params , im):

    if params.normalizeMethod == 'MinMax':
        im = np.float32(im)
        im = ( im-im.min() )/( im.max() - im.min() )
    return im

def normalizeMain(params , Input):

    if params.normalize:
        Input.Image = funcNormalize(params , Input.Image)

    return Input


