# Characterizing Robustness of Deep Neural Networks in Semantic Segmentation of Fluorescence Microscopy Images
  This repos provides an assay for quantitative evaluation of DNN models in segmentation of fluorescence microscopy images.
## Framework of our robustness evaluation
- **Dataset preparation:**   
We propose a synthetic scheme to generate simulated benchmark images of different corruption types (Mito-C, Nucleus-C, ER-C) to characterize the robustness of fluorescence segmentation. All of our datasets are publically shared at https://ieee-dataport.org/documents/robustness-benchmark-datasets-semantic-segmentation-fluorescence-images.

- **Model selection:**   
- **Quantification of robustness:**  
We define its robustness as the ratio between its IoU on degraded images and its IoU on the 'clean image' (We find empirically that FM images with an SNR of 8 to be sufficiently clean visually. Therefore, we take synthesized images with an SNR of 8 as our reference clean images).
## Dependencies
  - python 3
  - torch>=1.2.0
## How to use
