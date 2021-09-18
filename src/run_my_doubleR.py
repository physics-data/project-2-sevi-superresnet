from torchvision.models import resnet34, resnet18, resnet101, resnet50
from tqdm import tqdm
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt


class sevidataset(Dataset):
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
    print(np.mean(tmp),'|',np.sqrt(np.var(tmp)))


def loss_func(input, label):

    rate = 2/(np.array(range(1, 13))*4+1)
    rate = torch.Tensor(rate).cuda()
    delta = input - label
    return torch.sum(torch.sum((delta**2 * rate), dim=[1, 2]))/len(input)


def train(model, optimizer, dataloader, testdataloader, max_epochs=1, start_epoch=0):

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

        plt.plot(loss_list)
        tmp = np.array(loss_list)
        print(np.mean(tmp),'|',np.sqrt(np.var(tmp)))
        plt.savefig(f"pic/loss_pic{model_idx}/loss{epochs+start_epoch}.png")
        plt.show()
        plt.cla()

        eval(model, testdataloader)
        loss_list = []



inputs_train = []
labels_train = []
for i in [3,4,5,6,7,8,9,10]:#,10,11,12,13,14,15,16,17,18,19]:
    with h5py.File(f'data/dataset_my_double{i}.h5', 'r') as gen_data:
        inputs_train.append(np.array(gen_data['inputs']))
        labels_train.append(np.array(gen_data['labels']))
inputs_train = np.concatenate(inputs_train, axis=0)
labels_train = np.concatenate(labels_train, axis=0)
print(f'train_len: {len(inputs_train)}')

inputs_valid = []
labels_valid = []
for i in [0,1,2]:
    with h5py.File(f'data/dataset_my_double{i}.h5', 'r') as gen_data:
        # print(f'data/train{idx_of_gen_validdata}_ele_info.h5')
        inputs_valid.append(np.array(gen_data['inputs']))
        labels_valid.append(np.array(gen_data['labels']))
inputs_valid = np.concatenate(inputs_valid, axis=0)
labels_valid = np.concatenate(labels_valid, axis=0)
print(f'valid_len: {len(inputs_valid)}')

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
param_save_dir = f'param/resnet_my_double{model_idx}.pth'

checkpoint = torch.load(param_save_dir)#

model.load_state_dict(checkpoint['model'])#
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
optimizer.load_state_dict(checkpoint['optimizer'])#

max_epoch = 1
start_epoch = checkpoint['epoch']#
# start_epoch = 0#
end_epoch = start_epoch + max_epoch


dataset = sevidataset(inputs_train, labels_train)
validdataset = sevidataset(inputs_valid, labels_valid)
# dataset.cuda()
# validdataset.cuda()

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
validdataloader = DataLoader(validdataset, batch_size=5, shuffle=False)


train(model, optimizer, dataloader, validdataloader, max_epoch, start_epoch)
state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
         'epoch': end_epoch}
torch.save(state, param_save_dir)
# eval(model,dataloader)
# print(labels.count(0)/len(labels))