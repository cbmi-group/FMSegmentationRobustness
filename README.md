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
### 1. Standard train/test
Each model is trained on clean images with an SNR of 8. --model can be 'FCNs', 'UNet', 'SegNet', 'simple_unet', 'UNet_3', 'DeepLab', 'PSPNet' and 'ICNet'. The weights for each training  will be saved in 'checkpoints/model(for example, UNet)/save_path/'. The default batch size is 4.

    python Standard_Train.py --direction AtoB --train_dir data/mito/train/  --val_dir data/mito/val/  --save_pth trained_on_mito/ --gpu 1 --model UNet --norm std --lr 0.01 --epochs 500

Segmentation results  will be saved in 'results/model(for example, UNet)/load_path/test_dir/'. if '--best' is True, the best weights will be selected.

    python Standard_Test.py --direction AtoB --test_dir data/mito/test/ --load_pth trained_on_mito/ --gpu 0 --model UNet --norm std  --best True
### 2. Gaussian noise augmentation train/test
    python Gauss-Train.py --direction AtoB --train_dir data/mito/train/ --val_dir data/mito/val/  --save_pth Gauss_trained_on_mito/ --gpu 1 --model UNet --norm std --lr 0.01 --epochs 500
The test procedure is consistent with standard test, degraded images  will be loaded for testing.

    python Standard_Test.py --direction AtoB --test_dir data/mito/degraded_data/ --load_pth Gauss_trained_on_mito/ --gpu 0 --model UNet --norm std  --best True
### 3. Adversarial train/test
We apply PGD adversarial training, in which epsilon is 8/255, iteration number is 10, stpe size is 2/255 by default.
    
    python Adversarial_Train.py --direction AtoB --train_dir data/mito/train/  --val_dir data/mito/val/  --save_pth PGD_trained_on_mito/ --gpu 1 --model UNet --norm std --lr 0.01 --epochs 500
'--attack' can be 'FGSM', 'I-FGSM' or 'PGD'.

    python Adversarial_Test.py --direction AtoB  --load_pth PGD_trained_on_mito/ --test_dir data/mito/test/ --gpu 1 --model UNet --norm std  --best  True --attack I-FGSM
 
