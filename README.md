# SEVI 第二阶段大作业
[toc]
# 计算结果, 第一次提交

- 读取模型参数, 将上一阶段的结果归一化后乘以255作为输入
## 测试结果

- 初步测试结果为 **0.052**

# 识别测试集图片中的电子位置

- 根据高斯光斑特征还原电子打在MCP上的位置

- 将测试集转换为上阶段模型所需的输入图片
## 问题

- 处理50000张图片所需约4小时, 我们将数据分为10组, 分别处理后将结果相加

# 使用自己生成的训练集训练

- 使用自己生成的数据集进行训练

## 变化

- 进行了显卡内存优化
- 增大了batch_size
- 扩大了数据量, 生成近8000组训练集
- 实验发现resnet18模型的loss下降较稳定, 可能是增大了batch_size的原因, 所以换为resnet34进行训练, 希望得到更好效果

## 训练过程及结果

- resnet34 在50个epoch后即有明显下降
- 150个epoch后可以达到稳定, 我们定义的方式算出来的loss可以达到0.005
- trick:
    由于内存限制, 不能生成过大数据集. 所以我们在模型稳定后, 不断生成一组新的数据集(约400)替换原有的数据并再跑5个epoch, 直到200个epoch后, loss可以降为0.003



# 生成自己的数据集

## 数据范围
- 我们统计了第二类数据集中的数据范围

    - beta参数: 使用正态分布生成, 对于超出范围的数据进行截断, 对每个球壳生成前12个参数

        ```python
        # beta分布参数
        limit = [
            [0.35, 1],
            [0.43, 1.434],
            [0.25, 1.24],
            [0.118, 0.9],
            [0.045, 0.6],
            [0.012, 0.36],
            [0.0026, 0.2],
            [0.0004, 0.1],
            [7.0e-5, 0.045],
            [5.6e-6, 0.019],
            [1.2e-7, 0.0067],
            [2e-9, 0.0023]
        ]
        mu = [1,0.85,0.55,0.25,0.1,0.04,0.015,0.002,0.004,0,0,0]
        sigma = [0.65,0.5,0.5,0.5,0.5,0.3,0.2,0.1,0.05,0.02,0.01,0.01]
        ```
    - 半径

        $R$ 为 **0.3** 到 **0.9** 的均匀分布

        $\sigma_R$ 为 **0.005** 到 **0.009** 的均匀分布
        
- 将数据按半径大小排列, 生成**input-label**对


# 第二次尝试: 双球壳 v2.0

## 数据集
- `data/gen_pic_double.ipynb`文件,生成`data_double.h5`数据文件
- 将初步尝试中的电子位置投射在 $1024 \times 1024$ 的矩阵上, 矩阵每个位置表示该位置有多少个电子打在上面.

- 将两个但球壳矩阵相加, 除以数量总和归一化, 再乘以255, 便于观察图像.

- 生成了3000个数据, 将1/10用作验证集

## 模型变化

- 将最后fc层输出变为24

## 训练效果

- 训练过程较单球壳慢, 约100个epoch后, loss下降到0.01以下

- 效果和单球壳的结果一样好, 可以平均化相对误差达到7%, 证明该模型有效.

- 平均化相对误差:

    认为每个参数位置绝对误差相同, 根据平均loss计算得误差后, 除以0.7(与单球壳第一个个参数的数值相近)

    可以作为模型好坏的评价标准

- 局限性:

    1000个数据较少, 会导致模型泛化能力不够

    由于数据点为1000个单球壳的组合, 验证集中会出现训练集的数据, 导致模型过拟合,

    下一步尝试自己生成数据集

(模型训练过于复杂, 未完整保存过程)






# 第一次尝试 v1.0

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

