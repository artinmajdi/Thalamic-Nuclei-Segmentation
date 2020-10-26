import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import nibabel as nib
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means
import sys
from tqdm import tqdm
import os


def inputParameteres():
    # deafults:
    inputs = sys.argv
    init = {}

    if '--tv_chambolle' in inputs:
        init['denoising_method'] = 'tv_chambolle'

        # Denoising weight. The greater weight, the more denoising (at the expense of fidelity to input).
        string = 'weight'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 0.1

        # Relative difference of the value of the cost function that determines the stop criterion.
        # The algorithm stops when:  (E_(n-1) - E_n) < eps * E_0
        string = 'eps'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 0.0002

        # Maximal number of iterations used for the optimization.
        string = 'n_iter_max'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 200

    elif '--bilateral' in inputs:
        init['denoising_method'] = 'bilateral'

        # Standard deviation for grayvalue/color distance (radiometric similarity). A larger
        # value results in averaging of pixels with larger radiometric differences. Note, that
        # the image will be converted using the img_as_float function and thus the standard deviation
        # is in respect to the range [0, 1]. If the value is None the standard deviation of the image
        #  will be used.
        string = 'sigma_color'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 'None'

        # Standard deviation for range distance. A larger value results in averaging of pixels
        # with larger spatial differences.
        string = 'sigma_spatial'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 1

        # Window size for filtering. If win_size is not specified, it is calculated as
        # max(5, 2 * ceil(3 * sigma_spatial) + 1).
        string = 'win_size'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else max(5, 2 * np.ceil(
            3 * init['sigma_spatial']) + 1)

    elif '--wavelet' in inputs:
        init['denoising_method'] = 'wavelet'

        # The type of wavelet to perform and can be any of the options pywt.wavelist outputs.
        # The default is ‘db1’. For example, wavelet can be any of {'db2', 'haar', 'sym9'}
        # and many more.
        string = 'wavelet'
        init[string] = inputs[inputs.index(string) + 1] if string in inputs else 'db1'

        # Thresholding method to be used. The currently supported methods are { "BayesShrink" “VisuShrink” [2]. Defaults to “BayesShrink”}
        string = 'method'
        init[string] = inputs[inputs.index(string) + 1] if string in inputs else 'BayesShrink'

        # An optional argument to choose the type of denoising performed. It noted that choosing soft
        # thresholding given additive noise finds the best approximation of the original image. {‘soft’, ‘hard’}
        string = 'mode'
        init[string] = inputs[inputs.index(string) + 1] if string in inputs else 'soft'

    elif '--nl_means' in inputs:
        init['denoising_method'] = 'nl_means'

        # Size of patches used for denoising.
        string = 'patch_size'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 7

        # Maximal distance in pixels where to search patches used for denoising.
        string = 'patch_distance'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 11

        # Maximal distance in pixels where to search patches used for denoising.
        string = 'h'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 0.1

        # If True (default value), a fast version of the non-local means algorithm is used. If False,
        # the original version of non-local means is used. See the Notes section for more details about
        #  the algorithms.
        string = 'fast_mode'
        init[string] = inputs[inputs.index(string) + 1] if string in inputs else 'True'

        # The standard deviation of the (Gaussian) noise. If provided, a more robust computation of
        # patch weights is computed that takes the expected noise variance into account (see Notes below).
        string = 'sigma'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 0

    elif '--tv_bregman' in inputs:
        init['denoising_method'] = 'tv_bregman'

        # Maximal number of iterations used for the optimization.
        string = 'max_iter'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 100

        string = 'eps'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 0.001

        # Switch between isotropic and anisotropic TV denoising.
        string = 'isotropic'
        init[string] = inputs[inputs.index(string) + 1] if string in inputs else 'True'

        # Denoising weight. The smaller the weight, the more denoising (at the expense of less similarity
        # to the input). The regularization parameter lambda is chosen as 2 * weight.
        string = 'weight'
        init[string] = float(inputs[inputs.index(string) + 1]) if string in inputs else 0.1

    else:  # defaults
        init['denoising_method'] = 'wavelet'
        init['wavelet'] = 'db1'
        init['method'] = 'BayesShrink'
        init['mode'] = 'soft'

    init['inDir'] = sys.argv[1]
    init['outDir'] = sys.argv[2]

    return init


def normalizationFunc(im):
    params = {}
    params['max'] = im.max()
    params['min'] = im.min()
    im = (im - params['min']) / (params['max'] - params['min'])
    return im, params


def denormalizationFunc(im, params):
    return (im * (params['max'] - params['min'])) + params['min']


def denoisingFunc(init, noisy):
    # source: http://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_tv_bregman

    if init['denoising_method'] == 'tv_chambolle':
        denosied = denoise_tv_chambolle(noisy, weight=init['weight'], eps=init['eps'], n_iter_max=init['n_iter_max'])


    elif init['denoising_method'] == 'bilateral':

        print('Warning: this method is only for 2d images thus its looping over 3rd dimention')
        denosied = np.zeros(noisy.shape)
        for i in tqdm(range(noisy.shape[2])):
            denosied[..., i] = denoise_bilateral(noisy[..., i], sigma_color=init['sigma_color'],
                                                 sigma_spatial=init['sigma_spatial'], win_size=init['win_size'])


    elif init['denoising_method'] == 'wavelet':
        denosied = denoise_wavelet(noisy, wavelet=init['wavelet'], mode=init['mode'], method=init['method'])


    elif init['denoising_method'] == 'nl_means':
        denosied = denoise_nl_means(noisy, patch_size=init['patch_size'], patch_distance=init['patch_distance'],
                                    h=init['h'], fast_mode=init['fast_mode'], sigma=init['sigma'])


    elif init['denoising_method'] == 'tv_bregman':
        denosied = denoise_nl_means(noisy, weight=init['weight'], max_iter=init['max_iter'], eps=init['eps'],
                                    isotropic=init['isotropic'])

    denosied, _ = normalizationFunc(denosied)

    return denosied


def mkDir(directory):
    try:
        os.stat(directory)
    except:
        os.makedirs(directory)
    return directory


def writeFunc(denosied_dn, outDir, affine, header):
    Nifti1Image = nib.Nifti1Image(denosied_dn, affine)
    Nifti1Image.get_header = header
    # mkDir(outDir)
    nib.save(Nifti1Image, outDir)


mode = 'online'  # 'forTest'
init = testParameters() if mode == 'forTest' else inputParameteres()

im = nib.load(init['inDir'])
noisy, params = normalizationFunc(im.get_fdata())
denosied = denoisingFunc(init, noisy)
denosied_dn = denormalizationFunc(denosied, params)
writeFunc(denosied_dn, init['outDir'], im.affine, im.header)
