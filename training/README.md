# Training Code for KG-IQA: Knowledge-Guided Blind Image Quality Assessment with Few Training Samples
This is the training example of KG-IQA on the RBID dataset, which is small enough to re-train. The trainning process is the same for other datasets:

## 1. Data Prepareation

   To ensure high speed, save training images and lables, JND images, and NSS features into 'mat' files. The preparation process please refer to the published paper [KG-IQA](https://ieeexplore.ieee.org/document/10003665). The necessary 'mat' files can be downloaded from [Trainng files](https://pan.baidu.com/s/1EerM_rvNVo8Eevw74p3TNQ?pwd=z3oh). Please download these files and put them into the same folder of the training code.
   
## 2. Training the model

   Please 'run training_example_of_rbid_25percent.ipynb' to train the model.
   


If you like this work, please cite:

{
  author={Song, Tianshu and Li, Leida and Wu, Jinjian and Yang, Yuzhe and Li, Yaqian and Guo, Yandong and Shi, Guangming},
  
  journal={IEEE Transactions on Multimedia}, 
  
  title={Knowledge-Guided Blind Image Quality Assessment With Few Training Samples}, 
  
  year={Early Access,2022},
  
  doi={10.1109/TMM.2022.3233244}  
}


