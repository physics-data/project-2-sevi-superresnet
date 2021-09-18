from sevi_dataset import get_dataloader
from model import get_model, save_model
from train_func import train
import sys
import os


model_idx, max_epoch, train_batch_size, eval_batch_size = [
    int(i) for i in sys.argv[1:]]

if __name__ == '__main__':
    # 超参数
    learning_rate = 0.1

    # 读取训练集
    traindata_path = r'./data/traindata/'
    validdata_path = r'./data/validdata/'
    train_dataloader, valid_dataloader = get_dataloader(
        train_batch_size, eval_batch_size,  traindata_path, validdata_path)

    # 获取模型及加载训练参数

    param_save_dir = f'./param/'
    if not os.path.exists(param_save_dir):
        os.makedirs(param_save_dir)
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
