# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:18:18 2019

@author: Administrator
"""

import numpy as np
from scipy import io as sio
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import cv2
import os
import csv
import pandas as pd

# 读取csv至字典
#
ind_Y = (sio.loadmat('E:\\Database\\RBID\\RBID_mos.mat')['RBID_mos'][:, 0] - 1).astype('int')

all_lbs =sio.loadmat('E:\\Database\\RBID\\RBID_mos.mat')
name = np.array(all_lbs['RBID_str'][ind_Y])
num = len(name)

mos = np.array(all_lbs['RBID_mos'][:, 1:])

ind = np.arange(num)
np.random.seed(0)
np.random.shuffle(ind)
ind_train = ind[:int(len(ind) * 0.8)]
ind_test = ind[int(len(ind) * 0.8):]

imgs_all = np.zeros((num, 3, 244, 244), dtype=np.uint8)

impath = 'E:\\Database\\RBID\\Image'

for i in np.arange(0, num):
    im =plt.imread(impath + '\\' + name[i])

    # plt.imshow(im)
    # plt.show()
    imgs_all[i] = cv2.resize(im, (244, 244), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

sio.savemat('rbid_244.mat', {'train_name': name[ind_train],'X': imgs_all[ind_train], 'Y': mos[ind_train],
                               'test_name': name[ind_test],'Xtest': imgs_all[ind_test], 'Ytest': mos[ind_test], })

