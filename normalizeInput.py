import numpy as np

def normalizeInput(im , params):
    im = np.float32(im)
    return ( im-im.min() )/( im.max() - im.min() )
