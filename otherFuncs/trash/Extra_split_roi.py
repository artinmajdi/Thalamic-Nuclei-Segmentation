# import nilearn
import nibabel
import numpy as np
import os 


# directory = '/array/ssd/msmajdi/experiments/keras/exp9_test_Manoj/results/sE12_Cascade_FM00_Res_Unet2_NL3_LS_MyLogDice_US1_wLRScheduler_Main_Ps_ET_Init_3T_CV_a/2.5D_MV/vimp2_test_denoised_pre_surgery/'
dir = '/array/ssd/msmajdi/data/MS15T/vimp2_pre/Predictions_Left/'

def get_bounding_box(A):
    B = np.argwhere(A)
    start, stop = B.min(0), B.max(0) + 1
    return zip(start, stop)

def split_roi(roi, axis, split_axis):
    """
    Pages through the roi along axis and cuts each slice in half in the split_axis dimension.
    axis=None cuts the 3D bounding box in half.
    """
    first = np.zeros_like(roi)
    second = np.zeros_like(roi)
    def split_halves(roi, first, second, sl, axis):
        N = len(roi.shape)
        idx = [slice(sl, sl+1) if el is axis else slice(None) for el in xrange(N)]
        try:
            box = get_bounding_box(roi[idx])
        except ValueError:
            # No ROI in this slice
            return
        try:
            box[axis] = tuple(el+sl for el in box[axis])
        except TypeError:
            # Occurs for axis=None case
            pass
        first_idx = [slice(a, a + (b-a)/2) if i is split_axis else slice(a, b) for i, (a, b) in enumerate(box)]
        second_idx = [slice(a + (b-a)/2, b) if i is split_axis else slice(a, b) for i, (a, b) in enumerate(box)]
        # Try pasting in the original ROI to the half boxes
        try:
            first[first_idx] = roi[first_idx]
        except ValueError:
            # exception if half box is 0 along one dimension
            pass
        try:
            second[second_idx] = roi[second_idx]
        except ValueError:
            pass
    if axis is None:
        split_halves(roi, first, second, 0, axis)
    else:
        for sl in xrange(roi.shape[axis]):
            split_halves(roi, first, second, sl, axis)
    return first, second


input_nii = nibabel.load(dir + '6-VLP.nii.gz')
data = input_nii.get_data()
hdr = input_nii.get_header()
affine = input_nii.get_affine()

# Coronal axis for RL PA IS orientation
vlps = split_roi(data, None, 2)
for fname, sub_vlp in zip(['6_VLPv.nii.gz', '6_VLPd.nii.gz'], vlps):
    output_nii = nibabel.Nifti1Image(sub_vlp, affine, hdr)
    output_nii.to_filename(os.path.join(dir, fname))