import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import math
from scipy import io as sio
import torch.utils.data
import torchvision.models as models
from imgaug import augmenters as iaa
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import time
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from skimage import io
import os
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import torchvision.models as models
from functools import partial

from my_vision_transformer import VisionTransformer,_cfg
import matplotlib.pyplot as plt
import lmdb
from prefetch_generator import BackgroundGenerator

from torch.cuda.amp import autocast as autocast



class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Mydataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = torch.FloatTensor(labels)
    def __getitem__(self, index):
        return torch.from_numpy(self.imgs[index]),self.labels[index]
    def __len__(self):
        return (self.imgs).shape[0]

class Mydataset2(Dataset):
    def __init__(self, imgs, imgs2, labels, labels2):
        self.imgs = imgs
        self.imgs2 = imgs2
        self.labels = torch.FloatTensor(labels)
        self.labels2 = torch.FloatTensor(labels2)
    def __getitem__(self, index):
        return torch.from_numpy(self.imgs[index]), torch.from_numpy(self.imgs2[index]), self.labels[index], self.labels2[index]

    def __len__(self):
        return (self.imgs).shape[0]


def train(model, train_loader, optimizer, scaler, epoch, device, all_train_loss):
    model.train()
    st = time.time()
    op=[]
    tg=[]
    for batch_idx, (data, data2, target,brs_tg) in enumerate(train_loader):
        data, data2, target ,brs_tg= data.to(device), data2.to(device), target.to(device),brs_tg.to(device)
        torch.random.manual_seed(len(train_loader) * epoch + batch_idx)
        rd_ps = torch.randint(20, (3,))
        data = data[:, :, rd_ps[0]:rd_ps[0] + 224, rd_ps[1]:rd_ps[1] + 224]
        data2 = data2[:, :, rd_ps[0]:rd_ps[0] + 224, rd_ps[1]:rd_ps[1] + 224]
        if rd_ps[1] < 10:
            data = torch.flip(data, dims=[3])
            data2 = torch.flip(data2, dims=[3])
        data = data.float()
        data /= 255
        data[:, 0] -= 0.485
        data[:, 1] -= 0.456
        data[:, 2] -= 0.406
        data[:, 0] /= 0.229
        data[:, 1] /= 0.224
        data[:, 2] /= 0.225
        data2 = data2.float()
        data2 /= 255
        data2[:, 0] -= 0.485
        data2[:, 1] -= 0.456
        data2[:, 2] -= 0.406
        data2[:, 0] /= 0.229
        data2[:, 1] /= 0.224
        data2[:, 2] /= 0.225

        ind = np.arange(0, len(target))
        np.random.shuffle(ind)
        target2 = target[ind]

        target2 = target.clone()
        target2[target2 > 3.5] = target2[target2 > 3.5] - 1.5
        target2[target2 < 2.5] = target2[target2 < 2.5] + 1.5
        ind = torch.nonzero((target2 < 3.501) * (target2 > 2.499) == True)[:, 0]
        target2[ind[:int(len(ind) / 2)]] = target2[ind[:int(len(ind) / 2)]] - 1
        target2[ind[int(len(ind) / 2):]] = target2[ind[int(len(ind) / 2):]] + 1
        eps = torch.abs(data2 - data)
        data3=data.clone()
        data3.requires_grad = True
        model.eval()
        output,_,_,_ = model(data3)
        model.train()
        loss = F.mse_loss(output, target2)
        loss.backward()
        # print(pd.Series(output[:, 0].detach().cpu().numpy()).corr(pd.Series(target[:, 0].detach().cpu().numpy()), method="pearson"))
        data_grad = data3.grad.detach()
        data2 = data3.detach() + eps * (data_grad.sign())
        # model.eval()
        # output = model(data)
        # print(pd.Series(output[:, 0].detach().cpu().numpy()).corr((pd.Series(target[:, 0].detach().cpu().numpy())),method="pearson"))
        # print(pd.Series(output[:, 0].detach().cpu().numpy()).corr((pd.Series(target[:, 0].detach().cpu().numpy())),method="spearman"))

        data = torch.cat((data, data2), 0)
        ind_n = int(data.shape[0] / 2)
        optimizer.zero_grad()
        with autocast():
            output,ft,ft_tg,brs_ft = model(data)

            loss1 = F.mse_loss(output[:ind_n], target)
            loss2 = F.mse_loss(ft[:ind_n], ft_tg[ind_n:])
            loss3 = F.mse_loss(brs_ft[:ind_n], brs_tg)
            loss = 1*loss1 + 1.5* loss2+ 1* loss3
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        all_train_loss.append(loss.item())
        # loss.backward()
        # optimizer.step()

        op = np.concatenate((op, output[:ind_n, 0].detach().cpu().numpy()))
        tg = np.concatenate((tg, target[:, 0].cpu().numpy()))
        p1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson")
        s1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman")

        if batch_idx % 100 == 0:
            print('Train Epoch:{} [({:.0f}%)]\t Loss: {:.4f} Loss1: {:.4f} Loss2: {:.4f} Loss3: {:.4f} Pearson:{:.4f} Spearman:{:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item(), loss1.item(), loss2.item(), loss3.item(), p1, s1))

    print( 'Train ALL Pearson:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson"))
    print( 'Train  ALL Spearman:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman"))
    return all_train_loss


def test(model, test_loader, epoch, device, all_test_loss):
    model.eval()
    test_loss = 0
    pearson = 0
    spearman = 0
    op = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data ,target) in enumerate(test_loader):
            data,  target = data.to(device),target.to(device)
            data = data[:, :,10:10 + 224, 10:10 + 224]

            data = data.float()
            data /= 255
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225

            output,_,_ ,_= model(data)
            ind = torch.nonzero(target < 2)[:, 0]
            loss = F.mse_loss(output, target)
            all_test_loss.append(loss)
            test_loss += loss
            op = np.concatenate((op, output[:, 0].cpu().numpy()))
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))
            p1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson")
            s1 = pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman")
            pearson += p1
            spearman += s1
            if batch_idx % 100 == 0:
                print('Test Epoch:{} [({:.0f}%)]\t Loss: {:.4f}  Pearson:{:.4f} Spearman:{:.4f}'.format(
                    epoch, 100. * batch_idx / len(test_loader), loss.item(), p1, s1))

    test_loss /= (batch_idx + 1)
    pearson /= (batch_idx + 1)
    spearman /= (batch_idx + 1)
    print('Test : Loss:{:.4f} '.format(test_loss))
    print( 'ALL Pearson:', pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson"))
    print( 'ALL Spearman:',pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman"))
    return all_test_loss, pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="pearson"), pd.Series((op[::1])).corr((pd.Series(tg[::1])), method="spearman")



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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    device = torch.device("cuda")

    all_data = sio.loadmat('./rbid_244.mat')
    X = all_data['X']
    Y = all_data['Y']

    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    del all_data
    Y = Y * 0.8 + 1
    Ytest = Ytest * 0.8 + 1

    all_data = sio.loadmat('./rbid_jnd2_244.mat')
    X2 = all_data['X']
    Y2 = all_data['Y']
    Y2 = Y2 * 0.8 + 1
    Xtest2 = all_data['Xtest']
    Ytest2 = all_data['Ytest']
    del all_data

    all_data = sio.loadmat('./rbid224_brisque_feature.mat')
    Y2 = all_data['Y']
    Y2 = Y2 * 0.8 + 1
    Y2 = all_data['X_feat']

    Ytest2 = all_data['Ytest']
    Ytest2 = all_data['Xtest_feat']
    del all_data

    rt=0.25

    best_plccs=[]
    best_srccs = []
    best_low_plccs = []
    best_low_srccs = []
    for i in range(0,11):
        print('Split:',i)
        if i > 0:
            X = np.concatenate((X, Xtest), axis=0)
            Y = np.concatenate((Y, Ytest), axis=0)
            ind = np.arange(0, X.shape[0])
            np.random.seed(i)
            np.random.shuffle(ind)

            Xtest = X[ind[int(len(ind) * rt):]]
            Ytest = Y[ind[int(len(ind) * rt):]]
            X = X[ind[:int(len(ind) * rt)]]
            Y = Y[ind[:int(len(ind) * rt)]]

            X2 = np.concatenate((X2, Xtest2), axis=0)
            Y2 = np.concatenate((Y2, Ytest2), axis=0)
            ind = np.arange(0, X2.shape[0])
            np.random.seed(i)
            np.random.shuffle(ind)
        
            Xtest2 = X2[ind[int(len(ind) * rt):]]
            Ytest2 = Y2[ind[int(len(ind) * rt):]]
            X2 = X2[ind[:int(len(ind) * rt)]]
            Y2 = Y2[ind[:int(len(ind) * rt)]]
        if i<1:
            continue
        # if i>1:
        #     break
        ntimes=5
        best_plccs_ntimes=[]
        best_srccs_ntimes = []
        for n in range(ntimes):
            model1 = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model1.default_cfg = _cfg()
            model1.load_state_dict(torch.load("deit_small_patch16_224.pth")['model'])
            model2 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(384 *1, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, 1))

            model3 = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(384 * 1, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 384))

            model4 = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(384 * 1, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 36))

            model = MyModel(model1, model2, model3,model4)

            for param in model.parameters():
                param.requires_grad = False

            for param in model.model2.parameters():
                param.requires_grad = True
            for param in model.model3.parameters():
                param.requires_grad = True
            for param in model.model4.parameters():
                param.requires_grad = True
            model = nn.DataParallel(model.to(device))

            # model.load_state_dict(torch.load( 'koniq244_swintiny_adv_5split_'+str(i)+'.pt'))
            ###################################################################

            train_dataset = Mydataset2(X, X2, Y,Y2)
            test_dataset = Mydataset(Xtest, Ytest)

            max_plsp=-1
            min_loss = 1e8
            lr = 0.01
            weight_decay = 1e-3
            batch_size = 32*4
            epochs = 2000
            num_workers_train = 0
            num_workers_test = 0
            ct=0


            train_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_train,pin_memory=True)
            test_loader = DataLoaderX(test_dataset, batch_size=batch_size*4, shuffle=True, num_workers=num_workers_test,pin_memory=True)

            all_train_loss = []
            all_test_loss = []
            all_test_loss, _,_ = test(model, test_loader, -1, device, all_test_loss)
            ct = 0
            lr = 0.01
            max_plsp = -2
            scaler =  torch.cuda.amp.GradScaler()

            for epoch in range(epochs):
                print(lr)
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
                ct += 1
                start = time.time()
                all_train_loss = train(model, train_loader, optimizer, scaler, epoch, device, all_train_loss)
                print(time.time() - start)
                all_test_loss, plsp,_= test(model, test_loader, epoch, device, all_test_loss)
                print("time:", time.time() - start)
                if epoch == 80:
                    for param in model.parameters():
                        param.requires_grad = True
                    lr = 0.001

                if max_plsp < plsp:
                    save_nm = 'rbid224_deit_advandbrs_rt_25_withaug2_5split_'+str(i)+'times_'+str(n)+'.pt'
                    max_plsp = plsp
                    torch.save(model.state_dict(), save_nm)
                    ct = 0

                if epoch  ==160:
                    lr= 0.003
                if epoch == 240:
                    lr = 0.01
                    ct = 1

                if ct > 30 and epoch >240:
                    model.load_state_dict(torch.load(save_nm))
                    lr *= 0.3
                    ct = 0
                    if lr<1e-4:
                        all_test_loss, plsp, sp = test(model, test_loader, epoch, device, all_test_loss)
                        best_plccs_ntimes.append(plsp)
                        best_srccs_ntimes.append(sp)
                        print('Times:', n, 'End!', 'PLCC:', best_plccs_ntimes, 'SRCC:', best_srccs_ntimes)
                        break
            best_plccs.append(max(best_plccs_ntimes))
            best_srccs.append(max(best_srccs_ntimes))
        print('Split:', i, 'End!', 'PLCC:', best_plccs, 'SRCC:', best_srccs)
if __name__ == '__main__':
    main()

