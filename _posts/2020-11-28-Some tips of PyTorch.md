---
title: Some tips of PyTorch
date: 2020-11-28
tags:
- 深度学习框架
---
# 前言
在大三的时候已经使用PyTorch写过简单的DNN、CNN、预训练模型等，但当时只是被学分课(机器学习、计算机视觉)逼着写的，所以写完作业就基本不碰PyTorch了，也没有认真研究很多细节。现重新学习PyTorch，记录其很多重要但容易被忽略的细节，争取早日开始复现代码~

# 0 推荐学习资源
* [PYTORCH DOCUMENTATION](https://pytorch.org/docs/stable/index.html) - 官方文档
* [《动手学深度学习》-PyTorch](https://tangshusen.me/Dive-into-DL-PyTorch/#/) - 是 [Dive-into-DL](http://zh.d2l.ai/)([第二版](https://zh-v2.d2l.ai/index.html)) 的PyTorch中文重构版本
* [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) - 提供了很多非常简洁的模板代码，也很适合学习使用

# 1 经验细节汇总
## 1.1 数据操作
### 1.1.1 内存中的tensor
由于python对于变量在内存中的特殊储存方式，基于python的PyTorch也会因此受到影响，具体有以下几种形式：
1. 像Numpy一样，对一个tensor使用索引操作(如```new_tensor=tensor[1:]```)，索引出的结果与这个tensor**共享内存**(即修改一个，另一个也会跟着修改)
2. 用```.view()```改变tensor的形状，返回的新tensor与源tensor**共享内存**(顾名思义，```.view()```仅仅改变对该张量的观察角度，内部数据并未改变)。所以如果想返回一个**真正副本**，推荐使用```.clone.view()```或```.reshape()```
3. 使用```.numpy()```和```.from_numpy()```将tensor与Numpy中的array相互转换时，产生的tensor和array**共享内存**。如果这个tensor需要一个新的内存，那么可以使用```torch.tensor()```，这将消耗更多的时间和空间。

### 1.1.2 tensor的contiguous
顾名思义，**连续的**。PyTorch中张量的底层实现是使用C中的一维数组(一段连续的内存空间)，所以这里的连续是指在**内存中**是连续的。

使用```.view()```等方法时，必须先保证这个tensor是连续的(使用```.is_contiguous()```方法可以判断)。如果tensor在内存中不连续，则需要使用```.contiguous()```方法，他会**重新开辟一块内存空间**以保证连续。

> ```.is_contiguous()```的直观解释是**tensor底层一维数组元素的存储顺序与tensor按行优先一维展开的元素顺序是否一致**

PyTorch又提供了```.reshape()```方法，其实就等价于```.contiguous().view()```

下面是一个简单的示例：  
```python
a = torch.randn(16)
print(a.is_contiguous()) # True
b = a.view(-1, 4)
print(b.is_contiguous()) # True
c = b.transpose(0, 1)
print(c.is_contiguous()) # False

d = c.view(-1)
# Outputs:
RuntimeError: view size is not compatible with input tensor s size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

d = c.contiguous().view(-1)
print(d.shape)
# Outputs:
torch.Size([16])
```
了解更多，推荐阅读[参考资料[1]](https://zhuanlan.zhihu.com/p/64551412)

### 1.1.3 inplace操作
PyTorch操作inplace版本都有后缀_，代表就地修改，例如：
```python
y.add_(x)
y.copy_(x)
x.grad.data.zero_()
x.requires_grad_()
```

### 1.1.4 tensor在不同设备上移动
使用方法```.to(device)```可以将tensor在cpu和gpu之间相互移动。

## 1.2 网络结构
### 1.2.1 定义网络的几种方法
1. 继承```nn.Module```类，定义一些**层**以及```.forward()```方法，返回值为**输出**
2. 使用```nn.Sequential()```，按顺序地定义每一层

### 1.2.2 torch.nn的特性
1. 可使用```net.parameters()```来查看模型所有的可学习参数，返回一个生成器
2. ```torch.nn```仅支持**一个batch**样本的输入(不支持单样本)，如果只有单个样本，需要手动添加维度

### 1.2.3 train()与eval()
1. 在验证和测试时，需使用```model.eval()```方法，它可以自动关闭训练时使用的**Dropout**和**Batch Norm**

# 2 重要的库
## 2.1 TorchText
### 2.1.1 推荐资源
1. [TORCHTEXT DOCUMENTATION (0.8.1)](https://pytorch.org/text/0.8.1/) (官方文档，目前已更新到[0.9.0](https://pytorch.org/text/stable/index.html))
2. [pytorch/text](https://github.com/pytorch/text#data) (官方github仓库，其**readme**是一个非常简洁的使用指南)
3. [How to use TorchText for neural machine translation, plus hack to make it 5x faster](https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95#8a90) (一个优质的使用torchtext预处理机器翻译数据的教程)

### 2.1.2 使用技巧
在0.9.0版本中，之前版本的很多重要模块如```data```、```field```等已经移动到legacy中了，需要注意  
```
torchtext.legacy.data.field
torchtext.legacy.data.batch
torchtext.legacy.data.example
torchtext.legacy.data.iterator
torchtext.legacy.data.pipeline
torchtext.legacy.datasets
```

# 3 常用模板代码
## 3.1 模型的训练及验证
* 模型的训练  
```python
def train_epoch(epoch, model, optimizer, criterion, train_iter):
    model.train()
    for i, batch in enumerate(train_iter, start=1):
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, gold)
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            print('Epoch: {}, batch: [{}/{}], Loss: {:.5}'.format(epoch, i, len(train_iter), loss.item()))
```

* 模型的验证  
```python
def valid_epoch(epoch, model, optimizer, criterion, valid_iter):
    model.eval()
    with torch.no_grad():
        loss_list = []
        for _, batch in enumerate(valid_iter, start=1):
            out = model(inputs)
            loss = criterion(out, gold)
            loss_list.append(loss)
    return sum(loss_list) / len(valid_iter)
```

## 3.2 模型的保存和加载
* 模型的保存
```python
torch.save(model, PATH) # 方法1(不推荐)
torch.save(model.state_dict(), PATH) # 方法2
torch.save({'epoch':epoch, 'model':model.state_dict(), ...}, PATH) # 方法3
```

* 模型的加载
```python
# 对应方法1(不推荐)
model = torch.load(PATH)

# 对应方法2
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))

# 对应方法3
model = TheModelClass(*args, **kwargs)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model'])
```

**注意：** 在不同设备上保存或加载，需要添加```torch.load(PATH, map_location=device)```参数，且还需要使用```model.to(device)```。其中```device```是希望加载到的设备

# 参考资料
1. [PyTorch中的contiguous - 栩风](https://zhuanlan.zhihu.com/p/64551412)
2. [[TorchText]使用 - VanJordan](https://www.jianshu.com/p/e5adb235399e)
3. [torchtext入门教程，轻松玩转文本数据处理 - Lee](https://zhuanlan.zhihu.com/p/31139113)
4. [PyTorch 保存和加载模型 - 鑫鑫淼淼焱焱](https://zhuanlan.zhihu.com/p/82038049) ([原文 - Matthew Inkawhich](https://pytorch.org/tutorials/beginner/saving_loading_models.html))