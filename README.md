Thalamic Nuclei Analysis
=====================================
THIS IS A WORKING DOCUMENT

## Multi planar cascaded algorithm to segment thalamic nuclei.



#### Purpose: 
To develop a fast and accurate convolutional neural network based method for segmentation of thalamic nuclei.

#### Methods: 
A cascaded multi-planar scheme with a modified residual U-Net architecture was used to segment thalamic nuclei on conventional and white-matter-nulled (WMn) magnetization prepared rapid gradient echo (MPRAGE) data. A single network was optimized to work with images from healthy controls and patients with multiple sclerosis (MS) and essential tremor (ET), acquired at both 3T and 7T field strengths. WMn-MPRAGE images were manually delineated by a trained neuroradiologist using the Morel histological atlas as a guide to generate reference ground truth labels. Dice similarity coefficient and volume similarity index (VSI) were used to evaluate performance. Clinical utility was demonstrated by applying this method to study the effect of MS on thalamic nuclei atrophy. 

#### Results: 
Segmentation of each thalamus into twelve nuclei was achieved in under a minute. For 7T WMn-MPRAGE, the proposed method outperforms current state-of-the-art on patients with ET with statistically significant improvements in Dice for five nuclei (increase in the range of 0.05-0.18) and VSI for four nuclei (increase in the range of 0.05-0.19), while performing comparably for healthy and MS subjects. Dice and VSI achieved using 7T WMn-MPRAGE data are comparable to those using 3T WMn-MPRAGE data. For conventional MPRAGE, the proposed method shows a statistically significant Dice improvement in the range of 0.14-0.63 over FreeSurfer for all nuclei and disease types. Effect of noise on network performance shows robustness to images with SNR as low as half the baseline SNR.  Atrophy of four thalamic nuclei and whole thalamus was observed for MS patients compared to healthy control subjects, after controlling for the effect of parallel imaging, intracranial volume, gender, and age (p<0.004).

#### Conclusion: 
The proposed segmentation method is fast, accurate, performs well across disease types and field strengths, and shows great potential for improving our understanding of thalamic nuclei involvement in neurological diseases. 


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
