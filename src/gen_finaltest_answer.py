from torchvision.models import resnet34, resnet18, resnet101, resnet50
from tqdm import tqdm
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt

from sevi_dataset import get_dataloader
from model import get_model, save_model
from train_func import train
import sys
import os

model_idx, param_save_path = int(sys.argv[1]), sys.argv[2]

model_dict = {
    18: resnet18,
    34: resnet34,
    50: resnet50,
    101: resnet101
}

if __name__ == '__main__':

    # 获取模型
    model, optimizer, *_ = get_model(
        model_idx, param_save_path)

    model.eval()

    # 读取生成的数据
    b1 = np.load("data/finaltest_npy/final_0_5000_retry.npy")
    b2 = np.load("data/finaltest_npy/final_5000_10000_retry.npy")
    b3 = np.load("data/finaltest_npy/final_10000_15000_retry.npy")
    b4 = np.load("data/finaltest_npy/final_15000_20000_retry.npy")
    b5 = np.load("data/finaltest_npy/final_20000_25000_retry.npy")
    b6 = np.load("data/finaltest_npy/final_25000_30000_retry.npy")
    b7 = np.load("data/finaltest_npy/final_30000_35000_retry.npy")
    b8 = np.load("data/finaltest_npy/final_35000_40000_retry.npy")
    b9 = np.load("data/finaltest_npy/final_40000_45000_retry.npy")
    b10 = np.load("data/finaltest_npy/final_45000_50000_retry.npy")

    b = b1+b2+b3+b4+b5+b6+b7+b8+b9+b10
    # b = (b/2 + np.roll(b/2, -1, axis=0))
    b = torch.Tensor(b/b.sum()*255).cuda().unsqueeze(0).unsqueeze(0)
    #计算最终结果
    answer = model(b)
    print(answer)

    opt_answer = np.array([(0, 0.5, np.maximum(answer[0][0:12].cpu().detach().numpy(), 0)),
                      (1, 0.8, np.maximum(answer[0][12:24].cpu().detach().numpy(), 0))],
                     dtype=[('SphereId', 'u1'), ('R', '<f8'), ('beta', '<f8', (12,))])
    with h5py.File('final.h5', 'w') as out:
        out['Answer'] = opt_answer
