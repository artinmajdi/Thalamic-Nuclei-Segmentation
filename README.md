Thalamic Nuclei Analysis
=====================================

## Multi planar cascaded algorithm to segment thalamic nuclei.


### Installation and Requirements

Downloading the software
```
git clone https://github.com/artinmajdi/Thalamic-Nuclei-Segmentation.git
```

Installing the dependencies
```
conda env create -f requirements.yml
```

### Required Data Pre-Processing

* All data should be in the *.nii.gz format.

* In case of training or automatic measurement of Dice & VSI on test cases, the input ground-truth labels, ROI masks and other **images of each subject need to be co-registered** (per-subject, no need for inter-subject registration). 

* Image & labels within each subject should **have the same dimensions** (per subject, no need for whole database). This is, the number of voxels per dimension must be the same for all images of a subject. 

* Algorithm automatically resamples all images into the same voxel size

* Make sure that the **ground-truth labels** for training and evaluation represent the background with zero or FALSE. The system also assumes that the task’s classes are indexed as one or TRUE

### Prosecution Example
#### Training 

#### Testing

### Data Structure

### Citation
#### Link: <https://www.sciencedirect.com/science/article/pii/S0730725X20303118#t0005>

Majdi, M.S., Keerthivasan, M.B., Rutt, B.K., Zahr, N.M., Rodriguez, J.J. and Saranathan, M., 2020. 
<span style="color: green">  Automated thalamic nuclei segmentation using multi-planar cascaded convolutional neural networks. *Magnetic Resonance Imaging*.
