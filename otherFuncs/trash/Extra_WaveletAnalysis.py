import nibabel as nib
import numpy as np
import pywt
import matplotlib.pyplot as plt

# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# params = paramFunc.Run(UserInfo.__dict__, terminal=True)


a = nib.load('/array/ssd/msmajdi/data/preProcessed/CSFn_WMn/pre-steps/CSFn_Dataset1/cropped_Image/step1_registered_labels_croppedInput/vimp2_case1/crop_t1.nii.gz')
nib.viewers.OrthoSlicer3D(a.get_data()).show()


original = np.squeeze(a.slicer[:,:,50:51].get_data())
plt.imshow(original, cmap='gray')

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
