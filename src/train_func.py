from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path
import h5py


def loss_func(input, label):

    rate = 2/(np.array(range(1, 13))*4+1)
    rate = torch.Tensor(rate).cuda()
    delta = input - label
    return torch.sum(torch.sum(torch.sqrt(torch.sum((
        delta**2 * rate), dim=2)), dim=1), dim=0)/len(input)


def eval(model, dataloader):

    loss_list = []
    with torch.no_grad():
        model.eval()
        criterion = loss_func
        with tqdm(total=len(dataloader)) as t:
            t.set_description("@test")
            for step, batch in enumerate(dataloader):
                batch['inputs'] = batch['inputs'].cuda()
                batch['labels'] = batch['labels'].cuda()

                predict = model(batch['inputs'].unsqueeze(
                    1).cuda()).reshape(-1, 2, 12)
                labels = batch['labels'].cuda()

                loss = criterion(predict, labels)
                t.set_postfix(loss=loss.item())
                t.update(1)

                loss_list.append(loss.item())

                batch['inputs'] = batch['inputs'].cpu()
                batch['labels'] = batch['labels'].cpu()
    tmp = np.array(loss_list)
    print(np.mean(tmp), '|', np.sqrt(np.var(tmp)))

    # 保存loss数据
    loss_data_dir = './data/eval_loss/'
    loss_data_filename = 'eval_loss.h5'
    loss_data_path = loss_data_dir+loss_data_filename
    if not os.path.exists(loss_data_dir):
        os.makedirs(loss_data_dir)
    
    if Path(loss_data_path).is_file():
        # loss文件存在
        loss_file = h5py.File(loss_data_path,'r+')
        loss_file['mean'] = np.append(np.array(loss_file['mean'][:]),np.mean(tmp))
        loss_file['sigma'] = np.append(np.array(loss_file['sigma'][:]),np.sqrt(np.var(tmp)))
    else:
        loss_file = h5py.File(loss_data_path,'r+')
        loss_file['mean'] = np.array([np.mean(tmp)],dtype=np.float64)
        loss_file['sigma'] = np.array([np.sqrt(np.var(tmp))],dtype=np.float64)
    loss_file.close()
        


def train(model, model_idx, optimizer, dataloader, validdataloader, max_epochs=1, start_epoch=0):

    # loss_data_dir = './data/train_loss/'
    # loss_data_filename = 'tain_loss.h5'
    # loss_data_path = loss_data_dir+loss_data_filename
    # if not os.path.exists(loss_data_dir):
    #     os.makedirs(loss_data_dir)
    
    # if Path(loss_data_path).is_file():
    #     # loss文件存在
    #     loss_file = h5py.File(loss_data_path,'w+')
    #     loss_file_mean = loss_file['mean'][:]
    #     loss_file_sigma = loss_file['sigma'][:]
    # else:
    #     loss_file = h5py.File(loss_data_path,'w+')
    #     loss_file_mean = np.array([],dtype=np.float64)
    #     loss_file_sigma = np.array([],dtype=np.float64)
    loss_file_mean = np.array([],dtype=np.float64)
    loss_file_sigma = np.array([],dtype=np.float64)
    epochs = 0
    criterion = loss_func
    loss_list = []
    while (epochs < max_epochs):
        model.train()
        epochs += 1
        with tqdm(total=len(dataloader)) as t:
            for step, batch in enumerate(dataloader):
                batch['inputs'] = batch['inputs'].cuda()
                batch['labels'] = batch['labels'].cuda()
                t.set_description(
                    f"{model_idx},Epoch {epochs+start_epoch},{epochs}/{max_epochs}")

                predict = model(
                    batch['inputs'].unsqueeze(1)).reshape(-1, 2, 12)
                labels = batch['labels']

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
        loss_file_mean = np.append(loss_file_mean,np.mean(tmp))
        loss_file_sigma = np.append(loss_file_sigma,np.sqrt(np.var(tmp)))

        # plt.savefig(f"pic/loss_pic{model_idx}/loss{epochs+start_epoch}.png")
        # plt.show()
        # plt.cla()

        eval(model, validdataloader)
        loss_list = []

    loss_data_dir = './data/train_loss/'
    loss_data_filename = 'tain_loss.h5'
    loss_data_path = loss_data_dir+loss_data_filename
    if not os.path.exists(loss_data_dir):
        os.makedirs(loss_data_dir)
    
    if Path(loss_data_path).is_file():
        # loss文件存在
        loss_file = h5py.File(loss_data_path,'r+')
        loss_file['mean'] = np.append(np.array(loss_file['mean'][:]),loss_file_mean)
        loss_file['sigma'] = np.append(np.array(loss_file['sigma'][:]),loss_file_sigma)
    else:
        loss_file = h5py.File(loss_data_path,'r+')
        loss_file['mean'] = loss_file_mean
        loss_file['sigma'] = loss_file_sigma
    loss_file.close()