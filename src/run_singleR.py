from torchvision.models import resnet34, resnet18, resnet101
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
        criterion = nn.MSELoss()

        with tqdm(total=len(dataloader)) as t:
            for step, batch in enumerate(dataloader):

                t.set_description("{}".format('@test'))

                predict = model(batch['inputs'].unsqueeze(1))
                labels = batch['labels']

                loss = criterion(predict, labels)
                t.set_postfix(loss=loss.item())
                t.update(1)

                loss_list.append(loss.item())
    print(sum(loss_list)/len(loss_list))


def loss_func(input, label):

    rate = 2/(np.array(range(1, 13))*4+1)
    rate = torch.Tensor(rate).cuda()
    delta = input - label
    return torch.sum(torch.sum((delta**2 * rate), dim=1))/len(input)


def train(model, dataloader, testdataloader, max_epochs=1):

    epochs = 0
    criterion = loss_func
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_list = []
    while (epochs < max_epochs):
        model.train()
        tr_loss = 0.0
        global_step = 0
        epochs += 1

        with tqdm(total=len(dataloader)) as t:
            for step, batch in enumerate(dataloader):

                t.set_description("{},Epoch {}".format(model_idx, epochs))

                predict = model(batch['inputs'].unsqueeze(1))
                labels = batch['labels']

                loss = criterion(predict, labels)
                t.set_postfix(loss=loss.item())
                t.update(1)


                optimizer.zero_grad()
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()

        plt.plot(loss_list)
        print(sum(loss_list)/len(loss_list))
        plt.savefig(f"figures/loss_pic/loss_pic{model_idx}/loss{epochs}.png")
        plt.cla()

        eval(model, testdataloader)
        loss_list = []


idx_of_gen_traindata = 10
inputs = []
labels = []
for idx_of_gen_traindata in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
    with h5py.File(f'data/dataset_{idx_of_gen_traindata}.h5', 'r') as gen_data:
        inputs.append(np.array(gen_data['inputs']))
        labels.append(np.array(gen_data['labels'])[:, 0:12])
inputs = np.concatenate(inputs, axis=0)
labels = np.concatenate(labels, axis=0)
print(f'total_len: {len(inputs)}')


model_dict = {
    18: resnet18,
    34: resnet34,
    101: resnet101
}
model_idx = 18
model = model_dict[model_idx](pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                        bias=False)
model.fc = nn.Linear(512, 12)



model.load_state_dict(torch.load(f'param/resnet{model_idx}.pth'))
model.cuda()

dataset = sevidataset(inputs[0:900], labels[0:900])
testdataset = sevidataset(inputs[900:1000], labels[900:1000])
dataset.cuda()
testdataset.cuda()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
testdataloader = DataLoader(testdataset, batch_size=10, shuffle=False)

train(model, dataloader, testdataloader, 50)
torch.save(model.state_dict(), f'param/resnet{model_idx}.pth')
