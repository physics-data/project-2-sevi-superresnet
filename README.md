# SEVI 第二阶段大作业
[toc]
# 初步尝试

将第二类数据集的每个球壳(能级)的电子找出来, 复原为图像, 从图像中识别分布信息

先尝试从单球壳中求分布

## 数据名

原始数据为`train{num}.h5`, 将电子位置,半径,球壳的beta存为`dataset_{num}.h5`

单球壳图像:

![单球壳图像](figures/README_figures/gen_pic_single.png)

## 训练模型

使用`torchvision`的`resnet`模型, 包括18, 34, 101

## 模型

将resnet输入通道变为1, 最后线性层fc变为(512,12), 即最后输出为12个参数, 对应单球壳的beta分布前12个参数

## loss函数

loss函数定义如下:
```python
def loss_func(input,label):
    rate = 2/(np.array(range(1,13))*4+1)
    rate = torch.Tensor(rate)
    delta = input - label
    return torch.sum(torch.sum((delta**2 * rate),dim=1))/len(input)
```
即相比于评测函数, 这里少了求平方根的一步

两种评估对比较结果好坏是近乎相同的, 但这里计算量更小

## 训练方式

将第二类训练集的1000种数据分为训练集(900)和验证集(100)

由于训练过程较长, epoch较多, 训练过程并未保存完全, 大致情况为: 

resnet18: 

训练所需epoch较多, 在50个epoch之后loss偶有下降到0.01以下, 200个epoch后能下降到1e-5以下

resnet34,101:

100个epoch不能看出明显下降, 故放弃训练


## 设备信息

显卡: A100-SXM4-40GB 

Cpu: AMD EPYC 7302 16-Core Processor

训练时长: > 1min/epoch

## 评估

我认为单球壳的结果表明模型能提取到分布信息. 
根据loss结果, 平均到每个参数, 误差可以小于5%. 
下一步可以将两个球壳合并进行训练

