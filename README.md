Thalamic Nuclei Analysis
=====================================
THIS IS A WORKING DOCUMENT

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

### User Inputs
All user inputs can be modified inside the UserInfo.py python code inside the Parameters Folder

##### class experiment:
    exp_address        # Address to the experiment directory
    subexperiment_name # Subexperiment name
    train_address      # Path to the training data
    test_address       # Path to the testing data
    ReadAugments_Mode  # Reading augmented data. If TRUE, it'll read the data stored inside the subfolder called 'Augments'
    code_address       # Path to the code
    image_modality     # modality of the input data. wmn / csfn


##### class TestOnly:
    mode          # If TRUE , it will run the trained model on test cases.
    model_address # Address to the main folder holding the trained model.

##### class initialize:
    mode         # If TRUE, network weights will be initialized
    init_address # Path to the initialization network. If left empty, the algorithm will use the default path to sample initialization networks

##### class thalamic_side:    
    left # Running the network on left thalamus
    right # Running the network on right thalamus

##### class preprocess:
    """ Pre-processing flags
      - Mode             (boolean):   TRUE/FALSE
      - BiasCorrection   (boolean):   Bias Field Correction
      - Cropping         (boolean):   Cropping the input data using the cropped template
      - Reslicing        (boolean):   Re-slicing the input data into the same resolution
      - save_debug_files (boolean):   TRUE/FALSE
      - Normalize        (normalize): Data normalization
    """


### Terminal Commands
##### Terminal Assignments:
    - GPU index:          ('-g', '--gpu')
    - Image Modality:     ('-m', '--modality')
    - Train Directory:    ('--train')
    - Test  Directory:    ('--test')

##### Training: 
    Example: python main.py -g 3 --train <directory-to-train-cases-parent-folder> --test <directory-to-test-cases-parent-folder> --modality wmn

##### Testing
    Example: python main.py -g 3 --test "directory-to-test-cases-parent-folder" --modality csfn


### Data Structure
The address to train & test directories should be the parent directory that includes all train/test subjects(folders) (User should not point to the actual subject's folder).

##### Train & Test Directory Structure

    <Train-Directory>
         <subject 1>  (folder)
         <subejct 2>  (folder)
            ....
        
    <Test-Directory> 
         <subject m>   (folder)
         <subejct m+1> (folder)
            ....

##### Each Subject's folder structure "subject x"
Each subject should have its own folder consist of one *.nii.gz file representing the image and a sub-folder called Labels that includes all nifti labels named according to below

    image.nii.gz
    Labels (folder)
        1-THALAMUS.nii.gz 
        2-AV.nii.gz             
        4-VA.nii.gz  
        5-VLa.nii.gz   
        6-VLP.nii.gz       
        7-VPL.nii.gz  
        8-Pul.nii.gz  
        9-LGN.nii.gz       
        10-MGN.nii.gz      
        11-CM.nii.gz  
        12-MD-Pf.nii.gz    
        13-Hb.nii.gz    
        14-MTT.nii.gz 

### Citation
#### Link: <https://www.sciencedirect.com/science/article/pii/S0730725X20303118#t0005>
    
    Majdi, M.S., Keerthivasan, M.B., Rutt, B.K., Zahr, N.M., Rodriguez, J.J. and Saranathan, M., 2020. 
    Automated thalamic nuclei segmentation using multi-planar cascaded convolutional neural networks. 
    Magnetic Resonance Imaging.

### Abstract

#### Purpose: 
To develop a fast and accurate convolutional neural network based method for segmentation of thalamic nuclei.

#### Methods: 
A cascaded multi-planar scheme with a modified residual U-Net architecture was used to segment thalamic nuclei on conventional and white-matter-nulled (WMn) magnetization prepared rapid gradient echo (MPRAGE) data. A single network was optimized to work with images from healthy controls and patients with multiple sclerosis (MS) and essential tremor (ET), acquired at both 3T and 7T field strengths. WMn-MPRAGE images were manually delineated by a trained neuroradiologist using the Morel histological atlas as a guide to generate reference ground truth labels. Dice similarity coefficient and volume similarity index (VSI) were used to evaluate performance. Clinical utility was demonstrated by applying this method to study the effect of MS on thalamic nuclei atrophy. 

#### Results: 
Segmentation of each thalamus into twelve nuclei was achieved in under a minute. For 7T WMn-MPRAGE, the proposed method outperforms current state-of-the-art on patients with ET with statistically significant improvements in Dice for five nuclei (increase in the range of 0.05-0.18) and VSI for four nuclei (increase in the range of 0.05-0.19), while performing comparably for healthy and MS subjects. Dice and VSI achieved using 7T WMn-MPRAGE data are comparable to those using 3T WMn-MPRAGE data. For conventional MPRAGE, the proposed method shows a statistically significant Dice improvement in the range of 0.14-0.63 over FreeSurfer for all nuclei and disease types. Effect of noise on network performance shows robustness to images with SNR as low as half the baseline SNR.  Atrophy of four thalamic nuclei and whole thalamus was observed for MS patients compared to healthy control subjects, after controlling for the effect of parallel imaging, intracranial volume, gender, and age (p<0.004).

#### Conclusion: 
The proposed segmentation method is fast, accurate, performs well across disease types and field strengths, and shows great potential for improving our understanding of thalamic nuclei involvement in neurological diseases. 