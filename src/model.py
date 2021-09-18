from torchvision.models import resnet34, resnet18, resnet101, resnet50
import torch.nn as nn
import os
import torch

def get_model(model_idx,param_save_path,learning_rate=0.1,max_epoch=1):
    model_dict = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101
    }
    model = model_dict[model_idx](pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
    model.fc = nn.Linear(512, 24)

    if os.path.exists(param_save_path):
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



    return model,optimizer,start_epoch,end_epoch

def save_model(state, param_save_path):
    torch.save(state, param_save_path)