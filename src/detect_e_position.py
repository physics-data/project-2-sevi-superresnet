#!/usr/bin/env python

from tqdm import tqdm
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt

from sevi_dataset import get_dataloader, gauss_dataset
from model import get_model, save_model
from train_func import train
import sys
import os


class gauss_model(nn.Module):
    def __init__(self,):
        super(gauss_model, self).__init__()
        self.feature1 = 64
        self.feature2 = 128
        self.feature3 = 256
        self.feature4 = 256
        self.feature5 = 128
        self.feature6 = 1
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(1, self.feature1, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.feature1)

        self.conv2 = nn.Conv2d(self.feature1, self.feature2, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.feature2)

        self.conv3 = nn.Conv2d(self.feature2, self.feature3, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.feature3)

        self.conv4 = nn.Conv2d(self.feature3, self.feature4, kernel_size=2, stride=1, padding=1,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(self.feature4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.sigmoid(x)
        # print(x.shape)
        return x


def f1(label, predict):

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for idx, pred in enumerate(predict):
        if pred == 1:
            if label[idx] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if label[idx] == 0:
                TN += 1
            else:
                FN += 1
    return TP, FP, FN, TN


def eval(model, dataloader):

    predict_all = []
    labels_all = []

    with torch.no_grad():
        model.eval()
        # criterion = loss_func
        with tqdm(total=len(dataloader)) as t:
            t.set_description("@test")
            for step, batch in enumerate(dataloader):
                batch['inputs'] = batch['inputs'].cuda()
                batch['labels'] = batch['labels'].cuda()

                predict = model(batch['inputs'])
                labels = batch['labels']

                predict = torch.flatten(predict)

                predict_all.append(predict.detach().cpu())
                labels_all.append(labels.detach().cpu())

                # true += torch.sum(prediction == labels).item()
                # all += labels.shape[0]
                # loss = criterion(predict, labels)

                # t.set_postfix(loss=loss.item())
                t.update(1)

                # loss_list.append(loss.item())

                batch['inputs'] = batch['inputs'].cpu()
                batch['labels'] = batch['labels'].cpu()

    labels_all = torch.cat(labels_all)
    predict_all = torch.Tensor(
        [1 if x > 0.5 else 0 for x in torch.cat(predict_all)]).cuda()
    TP, FP, FN, TN = f1(labels_all, predict_all)
    print(f'|准确率{(TP+TN)/(TP+FP+FN+TN)}|0错误率{FP/(FP+TN):.3f}|1错误率{FN/(TP+FN):.3f}|')


def train(model, optimizer, dataloader, validdataloader, max_epochs=1, start_epoch=0):

    epochs = 0
    criterion = nn.BCELoss()
    loss_list = []
    while (epochs < max_epochs):
        loss_file_mean = np.array([], dtype=np.float64)
        loss_file_sigma = np.array([], dtype=np.float64)
        model.train()
        epochs += 1
        with tqdm(total=len(dataloader)) as t:
            for step, batch in enumerate(dataloader):
                batch['inputs'] = batch['inputs'].cuda()
                batch['labels'] = batch['labels'].cuda()
                t.set_description(
                    f"Gauss,Epoch {epochs+start_epoch},{epochs}/{max_epochs}")

                predict = model(batch['inputs'])
                labels = batch['labels'].unsqueeze(1)

                loss = criterion(predict, labels)
                t.set_postfix(loss=loss.item())
                t.update(1)

                optimizer.zero_grad()
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()

                batch['inputs'] = batch['inputs'].cpu()
                batch['labels'] = batch['labels'].cpu()

        # plt.plot(loss_list)
        tmp = np.array(loss_list)
        print(np.mean(tmp), '|', np.sqrt(np.var(tmp)))
        # loss_file_mean = np.append(loss_file_mean,np.mean(tmp))
        # loss_file_sigma = np.append(loss_file_sigma,np.sqrt(np.var(tmp)))

        eval(model, validdataloader)
        loss_list = []

# train_batch_size, valid_batch_size = sys.argv[1:]
# train_batch_size = int(train_batch_size)
# valid_batch_size = int(valid_batch_size)


train_batch_size, valid_batch_size, max_epoch = [int(i) for i in sys.argv[1:]]

learning_rate = 0.1


param_save_path = 'param/gauss_net.pth'
with h5py.File('data/gauss_validdata/gauss_validdata.h5', 'r') as validdata, \
        h5py.File('data/gauss_traindata/gauss_traindata.h5', 'r') as traindata:
    inputs_train = np.array(traindata['inputs'][:])
    labels_train = np.array(traindata['labels'][:])
    inputs_valid = np.array(validdata['inputs'][:])
    labels_valid = np.array(validdata['labels'][:])
    print(inputs_train[0].shape)
print(labels_valid.sum(), labels_valid.shape[0])
# 数据集
train_dataset = gauss_dataset(inputs_train, labels_train)
valid_dataset = gauss_dataset(inputs_valid, labels_valid)
train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=valid_batch_size, shuffle=False)

model = gauss_model()
if os.path.exists(param_save_path):
    print('loading param')
    checkpoint = torch.load(param_save_path)
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    end_epoch = start_epoch + max_epoch
else:
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    end_epoch = start_epoch + max_epoch


train(model, optimizer, train_dataloader,
      valid_dataloader, max_epoch, start_epoch)
# eval(model,valid_dataloader)
state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
         'epoch': end_epoch}
torch.save(state, param_save_path)
