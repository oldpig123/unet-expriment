The VerSe 2019 dataset includes 160 multidetector computed tomography (MDCT) image series of the spine from 141 patients. The dataset was prepared for the vertebral labeling and segmentation challenge "VerSe 2019" held at MICCAI 2019 in Shenzhen (https://verse2019.grand-challenge.org).

The CT scans are stored in NIfTI format. Data annotations that can be used for supervised learning include vertebra-level centroid coordinates (JSON format) and segmentation mask (NIFTI format). The data is divided into a set for training, development ('test'), and testing ('validation') purpose by a ratio of 50% / 25% / 25%. The validation set was released after an embargo period in December 2020.


Citation
--------
Please respect the patients whose data is presented here and our work. We spent >2 years for algorithmic development and >2000 working hours for manual corrections of segmentation masks.

By downloading this data you agreed to cite these papers in your work:

 1. Löffler M, Sekuboyina A, Jakob A, Grau AL, Scharr A, Husseini ME, Herbell M, Zimmer C, Baum T, Kirschke JS. A Vertebral Segmentation Dataset with Fracture Grading. Radiology: Artificial Intelligence, 2020 https://doi.org/10.1148/ryai.2020190138. 
 2. Liebl H, Schinz D, Sekuboyina A, ..., Kirschke JS. A computed tomography vertebral segmentation dataset with anatomical variations and multi-vendor scanner data Sci Data. 2021 Oct 28;8(1):284. doi: 10.1038/s41597-021-01060-0.
 3. Sekuboyina A, Bayat AH, Husseini ME, Löffler M, Menze BM, ..., Kirschke JS. VerSe: A Vertebrae labelling and segmentation benchmark for multi-detector CT images. Med Image Anal. 2021 Oct;73:102166. doi: 10.1016/j.media.2021.102166. Epub 2021 Jul 22. preliminary access at https://arxiv.org/abs/2001.09193

An overview of the data is provided in reference 1. The methods to generate the initial segmentations that were manually corrected afterwards are detailed in reference 2 and 3.

Please look for updates on citations and other relates publications on our website (https://deep-spine.de). Please note that you are required to cite the mentioned papers in any of your derivative work, both scientific (paper, abstract) and commercial (e.g. in a product description, web site).


Acknowledgements
----------------
This work has been supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 637164 — iBack — ERC-2014-STG).


License
-------
The data is published under the license CC BY-SA 4.0 (see licence.txt). When using the data you must cite the three papers mentioned above. Ethical approval to publish this data has been obtained from the local ethics committee at TUM (Proposal 27/19 S-SR).


Ethical approval
----------------
The local ethics committee at the Technical University Munich approved to publish this data (Proposal 27/19 S-SR).


Data structure
--------------
This dataset is available in two different ways:

1. Image series based (MICCAI): 160 image series of 141 patients are divided into a training (n=80), validation (n=40), and test (n=40) set as originally published for the MICCAI challenge.
2. Subject based: 141 patients holding 160 image series are divided into a training (n=67), validation (n=37), and test (n=37) set. Subject identifiers equal image series identifiers ('verseXXX'). For patient with 2 or 3 image series, new subject identifiers ('verse4XX') were introduced. This is an adaption of the Brain Imaging Data Structure (BIDS; https://bids.neuroimaging.io/).

We recommend using the subject based format, as this is consistent for the VerSe 2019 and VerSe 2020 datasets.


Files and file/directory names
------------------------------
- Image series based structure:

Four files can be found per image series in a directory named by the image series identifier ('verseXXX'):
(1) verse000.nii.gz - CT image series
(2) verse000_seg.nii.gz - segmentation mask
(3) verse000_ctd.json - centroid coordinates in 1mm isotropic ASL-space
(4) verse000_snapshot.png - Preview reformations of the annotated CT data.

Centroid coordinates (.json file) are given in 1mm isotropic resolution with coordinate origin at 'X'= anterior, 'Y'= superior, and 'Z'= left (ASL). 
'label' corresponds to the vertebral label:
1-7: cervical spine: C1-C7 
8-19: thoracic spine: T1-T12 
20-25: lumbar spine: L1-L6 
26,17: sacrum, cocygis - not labeled in this dataset 
28: additional 13th thoracic vertebra, T13, not present in this dataset.

- Subject based structure:

First, directories for CT images ('rawdata') and any derived data ('derivatives') were created. Then, subdirectories for each subject/patient were introduced. Directory names (=subject identifiers) are constructed using the original image series identifiers ('verseXXX'). New subject identifiers ('sub-verse4XX') were introduced for patient with 2 or 3 image series. The training, validation, and test sets hence include 67, 37 and 37 subject subdirectories, respectively. File names are constructed of entities, a suffix, and a file extension following the conventions of the Brain Imaging Data Structure (BIDS; https://bids.neuroimaging.io/). For patients with multiple image series, we included a 'split-<value>' entity, where <value> represents the original image series identifier.

Centroid coordinates of the subject based structure (.json file) are given in voxels in the image space. Vertebral labels ('label') are identical to those in image series based data structure.


Labeling rules
--------------
We only label 'free' vertebrae, i.e. we do not label the sacrum or transitional vertebrae that are (partly) fused with the sacrum. Such fused vertebrae are referred to as Castellvi grade 3 and 4. In this regard, all "free" vertebrae (including an ankylosis due to degeneration), are called lumbar. We consider L1 to be the first vertebra without ribs or with rib remnants smaller than 4cm on both sides in a horizontal alignment (including heterotopic ossification of the transverse process). The last thoracic vertebra should have at least one rib longer than 4cm in a typical diagonal downward alignment. In ambiguous cases, the shape of vertebra and facet joints is considered. If T1 is not present in the scan (i.e. visible within the scan's field-of-view), the thoracic spine is considered to have 12 vertebrae.
