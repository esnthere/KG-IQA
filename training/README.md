# Training Code for KG-IQA: Knowledge-Guided Blind Image Quality Assessment with Few Training Samples
This is the source code for 'KG-IQA: Knowledge-Guided Blind Image Quality Assessment with Few Training Samples'(doi:10.1109/TMM.2022.3233244).

1. Data Prepareation

   To ensure high speed, save training images and lables, JND images, and NSS features into 'mat' files. The preparation process please refer to the published paper [KG-IQA](https://ieeexplore.ieee.org/document/10003665). The mat files can be downloaded from [Trainng files](https://pan.baidu.com/s/1EerM_rvNVo8Eevw74p3TNQ?pwd=z3oh). 
   
2. Load pre-trained weight for test

   The models pre-trained on KonIQ-10k with 5%,10%,25%,80% samples are released. Each released model are obtained from the first split (the dataset are randomly splitted 10 times with numpy.random.seed(1)).


If you like this work, please cite:

{
  author={Song, Tianshu and Li, Leida and Wu, Jinjian and Yang, Yuzhe and Li, Yaqian and Guo, Yandong and Shi, Guangming},
  
  journal={IEEE Transactions on Multimedia}, 
  
  title={Knowledge-Guided Blind Image Quality Assessment With Few Training Samples}, 
  
  year={Early Access,2022},
  
  doi={10.1109/TMM.2022.3233244}  
}


