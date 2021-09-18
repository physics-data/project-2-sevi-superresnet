import h5py
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class sevi_dataset(Dataset):
    def __init__(self, inputs, labels):
        print("preparing dataset")
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels)
        assert len(self.inputs) == len(
            self.labels), "lens of inputs & labels are not same."
        self.tensor = {'inputs': self.inputs, 'labels': self.labels}
        print("finished")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {key: data[index] for key, data in self.tensor.items()}

    def cuda(self):
        for key in self.tensor:
            self.tensor[key] = self.tensor[key].cuda()

    def cpu(self):
        for key in self.tensor:
            self.tensor[key] = self.tensor[key].cpu()


def get_dataname(data_path=r'./data/traindata/'):
    assert os.path.exists(data_path), "训练集不存在"
    dataname = list(os.listdir(data_path))
    assert len(dataname) >= 3, "训练集不存在或过短"
    dataname.sort(reverse=False)
    return dataname


def get_dataset(data_path, train_data_names, valid_data_names):
    '''读取数据集'''

    inputs_train = []
    labels_train = []
    for dataname in train_data_names:
        with h5py.File(data_path+dataname, 'r') as gen_data:
            inputs_train.append(np.array(gen_data['inputs']))
            labels_train.append(np.array(gen_data['labels']))
    inputs_train = np.concatenate(inputs_train, axis=0)
    labels_train = np.concatenate(labels_train, axis=0)
    print(f'train_len: {len(inputs_train)}')

    inputs_valid = []
    labels_valid = []
    for dataname in valid_data_names:
        with h5py.File(data_path+dataname, 'r') as gen_data:
            inputs_valid.append(np.array(gen_data['inputs']))
            labels_valid.append(np.array(gen_data['labels']))
    inputs_valid = np.concatenate(inputs_valid, axis=0)
    labels_valid = np.concatenate(labels_valid, axis=0)
    print(f'valid_len: {len(inputs_valid)}')
    return inputs_train, labels_train, inputs_valid, labels_valid


def get_dataloader(train_batch_size=8, valid_batch_size=5, train_data_len=2, data_path=r'./data/traindata/'):

    data_names = get_dataname(data_path)
    inputs_train, labels_train, inputs_valid, labels_valid = get_dataset(
        data_path, data_names[0:train_data_len], data_names[train_data_len:])

    train_dataset = sevi_dataset(inputs_train, labels_train)
    valid_dataset = sevi_dataset(inputs_valid, labels_valid)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

    return train_dataloader,valid_dataloader
