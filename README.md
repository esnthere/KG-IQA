# KG-IQA: Knowledge-Guided Blind Image Quality Assessment with Few Training Samples
This is the source code for 'KG-IQA: Knowledge-Guided Blind Image Quality Assessment with Few Training Samples'(doi:10.1109/TMM.2022.3233244).

For test:
1. Data Prepareation

   To ensure high speed, save images and lables of each dataset with 'mat' files. Only need to run once.
   
2. Load pre-trained weight for test

   The models pre-trained on KonIQ-10k with 5%,10%,25%,80% samples are released. Each released model are obtained from the first split (the dataset are randomly splitted 10 times with numpy.random.seed(1)).

For train:



If you like this work, please cite:
{
  author={Song, Tianshu and Li, Leida and Wu, Jinjian and Yang, Yuzhe and Li, Yaqian and Guo, Yandong and Shi, Guangming},
  journal={IEEE Transactions on Multimedia}, 
  title={Knowledge-Guided Blind Image Quality Assessment With Few Training Samples}, 
  year={Early Access,2022},
  doi={10.1109/TMM.2022.3233244}}
}

