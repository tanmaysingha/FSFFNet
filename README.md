# MsFFNet
Multi-scale Feature Fusion Network (MsFFNet) is a semantic segmentation model for real-time computation. It reduces the large semantic gap among the rich feature maps by leveraging a multi-stage feature fusion technique at the decoder. For rich contextual representation, we deploy a feature scaling technique, similar to Atrous Spatial Pyramid Pooling (ASPP).
Model performance is evaluated by two public benchmarks- Cityscapes and BDD100K. Cityscapes provides 1024 * 2048 resolution fine-tune and coarse images, whereas BDD provides 720 * 1280 resolution fine-tune images. We utilize Cityscapes corase dataset to improve test set accuracy of the model. The proposed msFFNet can handle full resolution input images with less computational cost. To compare our model performance with other existing semantice segmentation models, we also trained FAST-SCNN, ContextNet, Bayesian SegNet, and DeepLab models. Our experiment exhibits that MSFFNet outperforms than these models and set a new state-of-the-art result on Cityscapes dataset. It produces 71.8% and 71% validation and test meanIoU respectively, while having only 1.3 million model parameters. On BDD100K dataset, model attains 55.2% validation meanIoU. This repository contains supplementary materials of the study. More details will be available upon acceptance of the paper. 

## Datasets
For this research work, we have used cityscapes benchmark datasets and CamVid dataset.
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
We trained our model with different input resolutions for cityscapes dataset. However, we mainly focus full resolution of cityscapes images. For Ca dataset, we use 768 x 1280px resolution altough original image size is 720 x 1280px. 

### DeepLab
![DeepLabV3+](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/DeepLab.png?raw=true)

### Bayesian SegNet
![Bayesian SegNet](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/bayes_segnet.png?raw=true)

### FAST-SCNN
![FAST-SCNN](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/fast_scnn.png?raw=true)
<b><u>IoU Over Classes on Validation Set</b></u>

### ContextNet
![ContextNet](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/separable_UNet.png?raw=true)

### FANet
![FANet](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/separable_UNet.png?raw=true)

### MsFFNet
![MsFFNet](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/separable_UNet.png?raw=true)

classes       |  IoU  |   nIoU
--------------|-------|---------
road          | 0.943 |    nan
sidewalk      | 0.761 |    nan
building      | 0.876 |    nan
wall          | 0.444 |    nan
fence         | 0.433 |    nan
pole          | 0.434 |    nan
traffic light | 0.511 |    nan
traffic sign  | 0.595 |    nan
vegetation    | 0.889 |    nan
terrain       | 0.546 |    nan
sky           | 0.908 |    nan
person        | 0.667 |  0.396
rider         | 0.437 |  0.228
car           | 0.899 |  0.787
truck         | 0.552 |  0.196
bus           | 0.650 |  0.365
train         | 0.451 |  0.197
motorcycle    | 0.395 |  0.186
bicycle       | 0.631 |  0.351
<b>Score Average | <b>0.633 | <b>0.338
 
 <b><u>IoU Over Categories </b></u>

categories    |  IoU   |  nIoU
--------------|--------|--------
flat          | 0.955  |   nan
construction  | 0.882  |   nan
object        | 0.529  |   nan
nature        | 0.891  |   nan
sky           | 0.908  |   nan
human         | 0.708  | 0.426
vehicle       | 0.878  | 0.756
<b>Score Average | <b>0.822  | <b>0.591
 
 <b><u>To see the performance of FAST-SCNN on test dataset, you can view the .csv file from here: </b></u>
 (https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/FAST_SCNN_Test_Results_Evaluated_By_Cityscapes_Server.csv)

### FANet
![FANet](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/FANet.png?raw=true)
<b><u>IoU Over Classes on Validation Set of Cityscapes</b></u>

classes       |  IoU  |   nIoU
--------------|-------|---------
road          | 0.962 |    nan
sidewalk      | 0.751 |    nan
building      | 0.893 |    nan
wall          | 0.527 |    nan
fence         | 0.473 |    nan
pole          | 0.470 |    nan
traffic light | 0.535 |    nan
traffic sign  | 0.646 |    nan
vegetation    | 0.898 |    nan
terrain       | 0.552 |    nan
sky           | 0.925 |    nan
person        | 0.702 |  0.459
rider         | 0.456 |  0.272
car           | 0.909 |  0.785
truck         | 0.470 |  0.201
bus           | 0.704 |  0.358
train         | 0.615 |  0.311
motorcycle    | 0.388 |  0.186
bicycle       | 0.652 |  0.403
<b>Score Average | <b>0.659 | <b>0.372

<b><u>IoU Over Categories on validation set of Cityscapes</b></u>

categories    |  IoU   |  nIoU
--------------|--------|--------
flat          | 0.967  |   nan
construction  | 0.894  |   nan
object        | 0.556  |   nan
nature        | 0.901  |   nan
sky           | 0.925  |   nan
human         | 0.715  | 0.489
vehicle       | 0.892  | 0.767
<b>Score Average | <b>0.836  | <b>0.628
 
 <b><u>To see the performance of FANet on test dataset, you can view the .csv file from here:</b></u>
  (https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/FANet_Test_Results_evaluated_by_Cityscapes_server.csv)

### Model prediction on CamVid dataset
![FANet_Vs_FAST_SCNN](https://github.com/tanmaysingha/FANet/blob/master/Predicted_images/CamVid_prediction.png?raw=true)

