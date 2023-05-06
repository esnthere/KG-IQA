# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
import torchvision.transforms.functional as tf
import os
import cv2
import numpy as np
from scipy import io as sio
import pandas as pd
from functools import partial
from my_vision_transformer import VisionTransformer,_cfg




class Mydataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = torch.FloatTensor(labels)

    def __getitem__(self, index):
        if self.imgs.shape[2]==244:
            img=self.imgs[index, :, 10:10 + 224, 10:10 + 224].transpose(1,2,0)
        else:
            img = self.imgs[index].transpose(1, 2, 0)
            
        img = tf.to_tensor(img)
        return img, self.labels[index]

    def __len__(self):
        return (self.imgs).shape[0]





def test(model, test_loader, epoch, device, all_test_loss):
    model.eval()
    op = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225
            data, target = data.to(device), target.to(device)
            output,_,_,_ = model(data)

            op = np.concatenate((op, output[:, 0].cpu().numpy()))
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))
    return op,tg



class MyModel(nn.Module):
    def __init__(self, model1,model2,model3,model4):
        super(MyModel, self).__init__()
        self.model1=model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def forward(self, data):
        xt= self.model1(data)
        x = self.model2(xt)
        xt2 = self.model3(xt)
        xt3 = self.model4(xt)
        return x,xt2,xt,xt3



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
    device = torch.device("cuda")

    model1 = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model1.default_cfg = _cfg()
    
    model2 = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(384 * 1, 64),
        nn.ReLU(),
        nn.Linear(64, 8),
        nn.ReLU(),
        nn.Linear(8, 1))

    model3 = nn.Sequential(
        nn.Linear(384 * 1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 384))

    model4 = nn.Sequential(
        nn.Linear(384 * 1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 36))

    model = MyModel(model1, model2, model3, model4)
    model = nn.DataParallel(model.to(device), device_ids=[0])

    model.load_state_dict(torch.load('koniq_25percent_JND_and_NSS.pt'))
    batch_size = 256
    num_workers_test = 0

#########################################################

    all_data = sio.loadmat('E:\Database\IQA Database\## KonIQ-10k Image Database\Koniq_224.mat')

    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    X = np.concatenate((X, Xtest), axis=0)
    Y = np.concatenate((Y, Ytest), axis=0)
    del all_data
    rt=0.10
    ind = np.arange(0, X.shape[0])
    np.random.seed(1)
    np.random.shuffle(ind)
    Xtest = X[ind[int(len(ind) * rt):]]
    Ytest = Y[ind[int(len(ind) * rt):]]


    test_dataset = Mydataset(Xtest, Ytest)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,
                              pin_memory=True)

    print("Koniq Test Results:")
    op, tg = test(model, test_loader, -1, device, [])
    print('ALL Pearson:', pd.Series(op).corr(pd.Series(tg), method="pearson"))
    print('ALL Spearman:', pd.Series(op).corr(pd.Series(tg), method="spearman"))

# # #####################################
    all_data = sio.loadmat('E:\Database\LIVEW\livew_224.mat')
    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)
    Y = Y / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    Ytest = Ytest / 25 + 1
    del all_data
    test_dataset=Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,
                              pin_memory=True)
    print("Livew Test Results:")

    op,tg=test(model,test_loader, -1, device, [])
    print('ALL Pearson:', pd.Series(op).corr(pd.Series(tg), method="pearson"))
    print('ALL Spearman:', pd.Series(op).corr(pd.Series(tg), method="spearman"))

#     ######################################################

    all_data = sio.loadmat('E:\Database\CID2013\\cid_224.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = (Y ) / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = (Ytest ) / 25 + 1
    del all_data

    test_dataset = Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,pin_memory=True)

    print("CID Test Results:")
    op, tg = test(model, test_loader, -1, device, [])
    print('ALL Pearson:', pd.Series(op).corr(pd.Series(tg), method="pearson"))
    print('ALL Spearman:', pd.Series(op).corr(pd.Series(tg), method="spearman"))

    #     #######################################################
#
    all_data = sio.loadmat('E:\Database\SPAQ\spaq_224.mat')
    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)
    Y = Y / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    Ytest = Ytest / 25 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,pin_memory=True)

    print("SPAQ Test Results:")
    op, tg = test(model, test_loader, -1, device, [])
    print('ALL Pearson:', pd.Series(op).corr(pd.Series(tg), method="pearson"))
    print('ALL Spearman:', pd.Series(op).corr(pd.Series(tg), method="spearman"))

    #
#     #######################################################
#
    all_data = sio.loadmat('E:\Database\RBID\\rbid_224.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = Y.reshape(Y.shape[0], 1)
    Y = Y * 0.8 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = Ytest.reshape(Ytest.shape[0], 1)
    Ytest = Ytest * 0.8 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,
                              pin_memory=True)

    print("RBID Test Results:")
    op, tg = test(model, test_loader, -1, device, [])
    print('ALL Pearson:', pd.Series(op).corr(pd.Series(tg), method="pearson"))
    print('ALL Spearman:', pd.Series(op).corr(pd.Series(tg), method="spearman"))


if __name__ == '__main__':
    main()

