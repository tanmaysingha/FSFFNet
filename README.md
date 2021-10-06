# FSFFNet
Feature Scaling Feature Fusion Network (FSFFNet) is a semantic segmentation model for real-time computation. It reduces the large semantic gap among the rich feature maps by leveraging a multi-stage feature fusion technique at the decoder. For rich contextual representation, we deploy a feature scaling technique, similar to Atrous Spatial Pyramid Pooling (ASPP).
Model performance is evaluated by two public benchmarks- Cityscapes and BDD100K. Cityscapes provides 1024 * 2048 resolution fine-tune and coarse images, whereas BDD provides 720 * 1280 resolution fine-tune images. We utilize Cityscapes corase dataset to improve test set accuracy of the model. The proposed msFFNet can handle full resolution input images with less computational cost. To compare our model performance with other existing semantice segmentation models, we also trained FAST-SCNN, ContextNet, Bayesian SegNet, and DeepLab models. Our experiment exhibits that MSFFNet outperforms than these models and set a new state-of-the-art result on Cityscapes dataset. It produces 71.8% and 69.4% validation and test meanIoU respectively, while having only 1.3 million model parameters. On BDD100K dataset, model attains 55.2% validation meanIoU. This repository contains supplementary materials of the study. More details will be available upon acceptance of the paper. 

### Complete pipeline of SCMNet
![pipeline](https://github.com/tanmaysingha/FSFFNet/blob/main/Prediction_samples/complete_pipeline.png?raw=true)

## Datasets
For this research work, we have used two publicly available benchmarks- Cityscapes and BDD100K datasets.
* Cityscapes - To access this benchmark, user needs an account. https://www.cityscapes-dataset.com/downloads/     
* BDD100K - To access this benchmark, visit this link: https://bair.berkeley.edu/blog/2018/05/30/bdd/

For cityscapes and BDD100K datasets, we use 19 classes to train and evaluate the model performance. Classes of BDD100K dataset are compatiable with Cityscapes dataset, although it provides total 41 class levels in compare to 35 classes of Cityscapes dataset. The class mapping between these two datasets are shown in the following table. BDD100k dataset is more challenging than Cityscapes. Therefore, we use transfter learning technique to improve model performance on both datasets. In the following table, classes highlighted by 255 TrainId are ignored classes.

<b><u>19 classes used for training</b></u>
Serial No. | TrainId | Cityscapes classes | TrainId | BDD100K classes   
-----------|---------|--------------------|---------|-----------------
    1      |    0    |        Road        |    0    |      Road
    2      |    1    |      Sidewalk      |    1    |    Sidewalk
    3      |    2    |      Building      |    2    |    Building
    4      |    3    |        Wall        |    3    |      Wall
    5      |    4    |       Fence        |    4    |     Fence
    6      |    5    |        Pole        |    5    |      Poll
    7      |    6    |   Traffic light    |    6    |  Traffic sign
    8      |    7    |   Traffic sign     |    7    |  Traffic sign
    9      |    8    |    Vegetation      |    8    |   Vegetation
   10      |    9    |      Terrain       |    9    |    Terrain
   11      |   10    |        Sky         |   10    |      Sky
   12      |   11    |      Person        |   11    |    Person
   13      |   12    |       Rider        |   12    |     Rider
   14      |   13    |        Car         |   13    |      Car
   15      |   14    |      Truck         |   14    |     Truck
   16      |   15    |        Bus         |   15    |      Bus
   17      |   16    |      Train         |   16    |     Train
   18      |   17    |    Motorcycle      |   17    |   Motorcycle
   19      |   18    |      Bicycle       |   18    |     Bicycle
   20      |   255   |    Unlabeled       |  255    |    Unlabeled
   21      |   255   |    Ego Vehicle     |  255    |   Ego Vehicle
   22      |   255   |Rectification Border|   -     |       -
   23      |   255   |    Out of roi      |   -     |       -
   24      |   255   |      Static        |  255    |     Static
   25      |   255   |     Dynamic        |  255    |     Dynamic
   26      |   255   |      Ground        |  255    |     Ground
   27      |   255   |     Parking        |  255    |    Parking
   28      |   255   |    Rail track      |  255    |   Rail track
   29      |   255   |    Guard rail      |  255    |   Guard rail
   30      |   255   |      Bridge        |  255    |     Bridge
   31      |   255   |      Tunnel        |  255    |     Tunnel
   32      |   255   |    Polegroup       |  255    |   Polegroup
   33      |   255   |     Caravan        |  255    |    Caravan
   34      |   255   |     Trailer        |  255    |    Trailer
   35      |   255   |  License plate     |         |
   36      |         |        -           |  255    |     Garage
   37      |         |        -           |  255    |     Banner
   38      |         |        -           |  255    |   Billboard  
   39      |         |        -           |  255    |  Lane divider
   40      |         |        -           |  255    |  Parking sign
   41      |         |        -           |  255    |  Street light
   42      |         |        -           |  255    |  Traffic cone
   43      |         |        -           |  255    |  Traffic device
   44      |         |                    |  255    | Trafic sign frame
  
 <b><u>Ignored classes from both datasets</b></u>
 
 TrainId | Cityscapes classes | TrainId | BDD100K classes   
---------|--------------------|---------|------------------
  255   |    Unlabeled       |  255    |    Unlabeled
  255   |    Ego Vehicle     |  255    |   Ego Vehicle
  255   |Rectification Border|   -     |       -
  255   |    Out of roi      |   -     |       -
  255   |      Static        |  255    |     Static
  255   |     Dynamic        |  255    |     Dynamic
  255   |      Ground        |  255    |     Ground
  255   |     Parking        |  255    |    Parking
  255   |    Rail track      |  255    |   Rail track
  255   |    Guard rail      |  255    |   Guard rail
  255   |      Bridge        |  255    |     Bridge
  255   |      Tunnel        |  255    |     Tunnel
  255   |    Polegroup       |  255    |   Polegroup
  255   |     Caravan        |  255    |    Caravan
  255   |     Trailer        |  255    |    Trailer
  255   |  License plate     |         |
        |        -           |  255    |     Garage
        |        -           |  255    |     Banner
        |        -           |  255    |   Billboard  
        |        -           |  255    |  Lane divider
        |        -           |  255    |  Parking sign
        |        -           |  255    |  Street light
        |        -           |  255    |  Traffic cone
        |        -           |  255    |  Traffic device
        |                    |  255    | Trafic sign frame

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
We trained our model with different input resolutions for different datasets. For Cityscapes, we train model with 1024 x 2048 px resolution whereas for BDD100K dataset, we use 768 x 1280 px resolution. Output produced by different models are displayed below.

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
![FSFFNet](https://github.com/tanmaysingha/MsFFNet/blob/main/Prediction_samples/FSFFNet.png?raw=true)
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

