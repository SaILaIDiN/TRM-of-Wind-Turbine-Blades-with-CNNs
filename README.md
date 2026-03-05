# TRM-of-Wind-Turbine-Blades-with-CNNs
This repository is part of the article "Explainable Machine Learning for Tower-Radar Monitoring of Wind Turbine Blades: Fine-Grained Blade Recognition Under Changing Operational Conditions" submitted to *MDPI Sensors*. 

The work investigates convolutional neural network (CNN)–based classification of wind turbine rotor blades using slow-time range radargrams acquired with an FMCW radar sensor mounted at the turbine tower. In addition to classification performance, the study analyzes the influence of environmental and operational conditions (EOCs), model architecture choices, data augmentation strategies, transfer learning, and explainability methods on robustness and generalization.

## Repository Overview
The codebase includes implementations for:
* Loading and preprocessing slow-time range radargrams
* Dataset partitioning by turbine, rotor blade, and EOCs
* CNN architectures, including a custom CNN and ResNet-18
* Training from scratch and with ImageNet-1k pretrained weights
* Evaluation using F1 scores and robustness/interpolation analyses
* Visualization of model attributions using Guided Grad-CAM

The repository is intended to support **reproducibility of the experimental findings** reported in the paper.

## Data Availability
Due to confidentiality agreements with the wind power plant operator, the entire FMCW radar dataset supporting the findings of this study cannot be publicly shared.  
A small subset of the data has been deposited in [Zenodo](https://zenodo.org/records/11483931).  
A related dataset of radargrams from simulated wind turbine blades can be found in [Zenodo](https://zenodo.org/records/13318595).

The repository provides:
* All preprocessing, training, evaluation, and visualization code
* Dataset interfaces and configuration files
* Fixed random seed handling to ensure reproducibility with equivalent datasets

Researchers with access to comparable radargram data can reproduce the experiments by following the documented data structure.

## Requirements
The code was developed and tested using:
* Python >= 3.10
* PyTorch 2.1.0
* CUDA 12.1
* timm 1.0.9
* albumentations 1.4.15
* Numpy
* matplotlib

All required packages are listed in [requirements.txt](requirements.txt).

## License
This paper is released under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

## Citation
If you use this code in your research, please cite the corresponding paper:

```bibtex
@Article{alipeksensors2026,
AUTHOR = {Alipek, Sercan and Kexel, Christian and Moll, Jochen},
TITLE = {Explainable Machine Learning for Tower-Radar Monitoring of Wind Turbine Blades: Fine-Grained Blade Recognition Under Changing Operational Conditions},
JOURNAL = {Sensors},
VOLUME = {26},
YEAR = {2026},
NUMBER = {4},
ARTICLE-NUMBER = {1083},
ISSN = {1424-8220},
DOI = {https://doi.org/10.3390/s26041083}
}
```

## Funding
This research was funded by the BMWK (German Federal Ministry for Economic Affairs and Climate Action) under the grant number 03EE2035A.
