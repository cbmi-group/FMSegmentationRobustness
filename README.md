# Characterizing Robustness of Deep Neural Networks in Semantic Segmentation of Fluorescence Microscopy Images
  This repos provides an assay for quantitative evaluation of DNN models in segmentation of fluorescence microscopy images.
## Framework of our robustness evaluation
- **Dataset preparation:**   
We propose a synthetic scheme to generate simulated benchmark images of different corruption types (Mito-C, Nucleus-C, ER-C) to characterize the robustness of fluorescence segmentation. All of our datasets are publically shared at https://ieee-dataport.org/documents/robustness-benchmark-datasets-semantic-segmentation-fluorescence-images.

<img src="https://user-images.githubusercontent.com/55579451/184861879-3fc0881f-662b-481a-b9a2-d868ec47ff47.png" width="50%">

- **Model selection:**  
We examine eight models, including FCN, SegNet, UNet, DeepLab, PSPNet, and ICNet, which have been validated extensively in the literature. UNet_3 and Sim_UNet are two simplified variants of UNet.
- **Quantification of robustness:**  
We define its robustness as the ratio between its IoU on degraded images and its IoU on the 'clean image' (We find empirically that FM images with an SNR of 8 to be sufficiently clean visually. Therefore, we take synthesized images with an SNR of 8 as our reference clean images).
## Dependencies
  - python 3
  - torch>=1.2.0
## How to use
