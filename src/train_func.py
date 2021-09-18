from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt


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


def train(model, model_idx, optimizer, dataloader, testdataloader, max_epochs=1, start_epoch=0):
    
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
        print(np.mean(tmp), '|', np.sqrt(np.var(tmp)))
        plt.savefig(f"pic/loss_pic{model_idx}/loss{epochs+start_epoch}.png")
        plt.show()
        plt.cla()

        eval(model, testdataloader)
        loss_list = []
