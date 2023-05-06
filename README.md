# KG-IQA: Knowledge-Guided Blind Image Quality Assessment with Few Training Samples
This is the source code for [KG-IQA: Knowledge-Guided Blind Image Quality Assessment with Few Training Samples](https://ieeexplore.ieee.org/document/10003665).

For test:
1. Data Prepareation

   To ensure high speed, save images and lables of each dataset with 'mat' files. Only need to run 'data_preparation_example_for_koniq.py' once for each dataset.
   
2. Load pre-trained weight for test

   The models pre-trained on KonIQ-10k with 5%,10%,25%,80% samples are released. The dataset are randomly splitted several times during training, and each released model is obtained from the first split ( numpy.random.seed(1)).
   
   The pre-trained models can be downloaded from: [Pre-trained models](https://pan.baidu.com/s/1kKGTp1iS0QGhuYGSJQVhTg?pwd=o80k). 
   
For train:

  The training code is available in the 'training' folder.


If you like this work, please cite:

{
  author={Song, Tianshu and Li, Leida and Wu, Jinjian and Yang, Yuzhe and Li, Yaqian and Guo, Yandong and Shi, Guangming},  
  journal={IEEE Transactions on Multimedia},   
  title={Knowledge-Guided Blind Image Quality Assessment With Few Training Samples},   
  year={Early Access,2022},  
  doi={10.1109/TMM.2022.3233244}  
}

