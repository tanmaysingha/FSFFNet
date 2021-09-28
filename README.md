# FSFFNet
Feature Scaling Feature Fusion Network (FSFFNet) is a semantic segmentation model for real-time computation. It reduces the large semantic gap among the rich feature maps by leveraging a multi-stage feature fusion technique at the decoder. For rich contextual representation, we deploy a feature scaling technique, similar to Atrous Spatial Pyramid Pooling (ASPP).
Model performance is evaluated by two public benchmarks- Cityscapes and BDD100K. Cityscapes provides 1024 * 2048 resolution fine-tune and coarse images, whereas BDD provides 720 * 1280 resolution fine-tune images. We utilize Cityscapes corase dataset to improve test set accuracy of the model. The proposed msFFNet can handle full resolution input images with less computational cost. To compare our model performance with other existing semantice segmentation models, we also trained FAST-SCNN, ContextNet, Bayesian SegNet, and DeepLab models. Our experiment exhibits that MSFFNet outperforms than these models and set a new state-of-the-art result on Cityscapes dataset. It produces 71.8% and 69.4% validation and test meanIoU respectively, while having only 1.3 million model parameters. On BDD100K dataset, model attains 55.2% validation meanIoU. This repository contains supplementary materials of the study. More details will be available upon acceptance of the paper. 

## Datasets
For this research work, we have used two publicly available benchmarks- Cityscapes and BDD100K datasets.
* Cityscapes - To access this benchmark, user needs an account. https://www.cityscapes-dataset.com/downloads/     
* BDD100K - To access this benchmark, visit this link: https://bair.berkeley.edu/blog/2018/05/30/bdd/

## Metrics
To understand the metrics used for model performance evaluation, please  refer here: https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results

## Transfer Learning
For performance comparison, we trained few off-line and real-time existing models under same configuration and compared their performance with the proposed model. Some existing models require the use of ImageNet pretrained models to initialize their weights. Details will be given soon.

## Requirements for Project
* TensorFlow 2.1
  * This requires CUDA >= 10.1
  * TensorRT will require an NVIDIA GPU with Tensor Cores.
  * Horovod framework (for effective utilization of resources and speed up GPUs)
* Keras 2.3.1
* Python >= 3.7

## Results
We trained our model with different input resolutions for cityscapes dataset. However, we mainly focus on full resolution of cityscapes images. For BDD100K dataset, we use 768 x 1280px resolution altough original image size is 720 x 1280px. 

### Complete pipeline of SCMNet
![pipeline](https://github.com/tanmaysingha/FSFFNet/blob/main/Prediction_samples/complete_pipeline.png?raw=true)

### DeepLab
![DeepLabV3+](https://github.com/tanmaysingha/MsFFNet/blob/main/Prediction_samples/Deeplab.png?raw=true)

### Bayesian SegNet
![Bayesian SegNet](https://github.com/tanmaysingha/MsFFNet/blob/main/Prediction_samples/Bayes_SegNet.png?raw=true)

### FAST-SCNN
![FAST-SCNN](https://github.com/tanmaysingha/MsFFNet/blob/main/Prediction_samples/fast_scnn.png?raw=true)

### ContextNet
![ContextNet](https://github.com/tanmaysingha/MsFFNet/blob/main/Prediction_samples/ContextNet.png?raw=true)

### FANet
![FANet](https://github.com/tanmaysingha/MsFFNet/blob/main/Prediction_samples/FANet.png?raw=true)

### FSFFNet
![FSFFNet](https://github.com/tanmaysingha/MsFFNet/blob/main/Prediction_samples/MsFFNet.png?raw=true)
#### Class and Categorywise model performance on Cityscapes Validation set
![val_meanIoU](https://github.com/tanmaysingha/MsFFNet/blob/main/Prediction_samples/best_val_meanIoU.png?raw=true) 
 
 <b><u>To see the performance of FSFFNet on test dataset, you can view the following file from here. Upno acceptance of paper, test result will be published in Cityscapes leaderboard. </b></u>
 (https://github.com/tanmaysingha/FSFFNet/blob/main/Prediction_samples/Cityscapes_Test_results.pdf)

 ### FSFFNet prediction on Cityscapes test images
![Cityscapes_test_set](https://github.com/tanmaysingha/MsFFNet/blob/main/Prediction_samples/Cityscapes_test.png?raw=true) 
 
### Model prediction on BDD100K dataset
#### Validation set
![BDD100K_val_set](https://github.com/tanmaysingha/FSFFNet/blob/main/Prediction_samples/BDD_val_predictions.png?raw=true)
#### Test set
![BDD100K_test_set](https://github.com/tanmaysingha/MsFFNet/blob/main/Prediction_samples/BDD100K_test.png?raw=true)

