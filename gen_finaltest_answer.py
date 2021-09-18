from torchvision.models import resnet34, resnet18, resnet101, resnet50
from tqdm import tqdm
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt

model_dict = {
    18: resnet18,
    34: resnet34,
    50: resnet50,
    101: resnet101
}

model_idx = 34

model = model_dict[model_idx](pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                        bias=False)
model.fc = nn.Linear(512, 24)


# 读取训练数据
param_save_dir = f'param/resnet_my_double34.pth'

checkpoint = torch.load(param_save_dir)#

model.load_state_dict(checkpoint['model'])#
model.cuda()
model.eval()
# b = np.load("final.npy")
b1 = np.load("finaltest_npy/final_0_5000.npy")
b2 = np.load("finaltest_npy/final_5000_10000.npy")
b3 = np.load("finaltest_npy/final_10000_15000.npy")
b4 = np.load("finaltest_npy/final_15000_20000.npy")
b5 = np.load("finaltest_npy/final_20000_25000.npy")
b6 = np.load("finaltest_npy/final_25000_30000.npy")
b7 = np.load("finaltest_npy/final_30000_35000.npy")
b8 = np.load("finaltest_npy/final_35000_40000.npy")
b9 = np.load("finaltest_npy/final_40000_45000.npy")
b10 = np.load("finaltest_npy/final_45000_50000.npy")

b = b1+b2+b3+b4+b5+b6+b7+b8+b9+b10

b = (b/2 + np.roll(b/2, -1, axis=0))



b = torch.Tensor(b/b.sum()*255).cuda().unsqueeze(0).unsqueeze(0)

print(model(b))