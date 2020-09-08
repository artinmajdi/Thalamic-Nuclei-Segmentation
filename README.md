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

* Input ground-truth labels and the image for each subject should be co-registered** (per-subject, no need for inter-subject registration). 

* Algorithm automatically resamples & normalizes all images into the same voxel size and dynamic range

* Make sure that the **ground-truth labels** for training and evaluation represent the background with zero/FALSE and foreground with one/TRUE.

### Prosecution Example
#### Training 

#### Testing

### Data Structure

### Citation
#### Link: <https://www.sciencedirect.com/science/article/pii/S0730725X20303118#t0005>

Majdi, M.S., Keerthivasan, M.B., Rutt, B.K., Zahr, N.M., Rodriguez, J.J. and Saranathan, M., 2020. 
<span style="color: green">  Automated thalamic nuclei segmentation using multi-planar cascaded convolutional neural networks. *Magnetic Resonance Imaging*.
