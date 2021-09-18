
# from torch.optim import optimizer
# from tqdm import tqdm
# import h5py
# import numpy as np
# import torch
# import torch.nn as nn

# from torch.nn import functional as F
# import matplotlib.pyplot as plt
from sevi_dataset import get_dataloader
from model import get_model,save_model
from train_func import train
import sys


model_idx, max_epoch, train_batch_size, eval_batch_size = [
    int(i) for i in sys.argv[1:]]

if __name__ == '__main__':
    # 超参数
    learning_rate = 0.5

    # 读取训练集
    data_path = r'./data/traindata/'
    train_dataloader, valid_dataloader = get_dataloader(8, 1, 2, data_path)

    # 获取模型及加载训练参数

    param_save_dir = f'./param/'
    param_save_name = f'resnet_my_double{model_idx}.pth'
    param_save_path = param_save_dir+param_save_name
    model, optimizer, start_epoch, end_epoch = get_model(
        model_idx, param_save_path, learning_rate=learning_rate, max_epoch=max_epoch)

    # 开始训练
    train(model, model_idx, optimizer, train_dataloader,
          valid_dataloader, max_epoch, start_epoch)

    # 保存数据
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'epoch': end_epoch}
    save_model(state, param_save_path)

