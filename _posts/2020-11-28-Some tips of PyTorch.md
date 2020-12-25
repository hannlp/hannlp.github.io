---
title: Some tips of PyTorch
date: 2020-11-28
tags:
- 深度学习框架
---
# 前言
在大三的时候已经使用PyTorch写过简单的DNN、CNN、预训练模型等，但当时只是被学分课(机器学习、计算机视觉)逼着写的，所以写完作业就基本不碰PyTorch了，也没有认真研究很多细节。现重新学习PyTorch，记录其很多重要但容易被忽略的细节，争取早日开始复现代码~

# 0 推荐学习资源
* [PYTORCH DOCUMENTATION](https://pytorch.org/docs/stable/index.html) - 最好的学习资源当然是官方文档啦
* [《动手学深度学习》-PyTorch](https://tangshusen.me/Dive-into-DL-PyTorch/#/) - 只需了解基础的线性代数、微分和概率，以及基础的Python编程，即可迅速入门PyTorch。是 [Dive-into-DL](http://zh.d2l.ai/) 的PyTorch重构版本(中文)
* [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) - 提供了很多非常简洁的模板代码，也很适合学习使用

# 1 数据操作
## 1.1 内存中的tensor
由于python对于变量在内存中的特殊储存方式，基于python的PyTorch也会因此受到影响，具体有以下几种形式：
1. 像Numpy一样，对一个tensor使用索引操作(如```new_tensor=tensor[1:]```)，索引出的结果与这个tensor**共享内存**(即修改一个，另一个也会跟着修改)
2. 用```.view()```改变tensor的形状，返回的新tensor与源tensor**共享内存**(顾名思义，```.view()```仅仅改变对该张量的观察角度，内部数据并未改变)。所以如果想返回一个**真正副本**，推荐使用```.clone.view()```或```.reshape()```
3. 使用```.numpy()```和```.from_numpy()```将tensor与Numpy中的array相互转换时，产生的tensor和array**共享内存**。如果这个tensor需要一个新的内存，那么可以使用```torch.tensor()```，这将消耗更多的时间和空间。

## 1.2 tensor的contiguous
顾名思义，**连续的**。这里的连续是指在**内存中**是连续的。PyTorch中张量的底层实现是使用C中的一维数组(一段连续的内存空间)

使用```.view()```等方法时，必须先保证这个tensor是连续的。使用```.is_contiguous()```方法可以判断。  
> ```.is_contiguous()```的直观解释是**tensor底层一维数组元素的存储顺序与tensor按行优先一维展开的元素顺序是否一致**

如果tensor在内存中不连续，则需要使用```.contiguous()```方法，他会**重新开辟一块内存空间**以保证连续。

PyTorch又提供了```.reshape()```方法，其实就等价于```.contiguous().view()```

## 1.3 inplace操作
PyTorch操作inplace版本都有后缀_，代表就地修改，例如：
```python
y.add_(x)
y.copy_(x)
x.grad.data.zero_()
x.requires_grad_()
```

## 1.4 tensor在cpu或gpu
使用方法```.to()```可以将tensor在cpu和gpu之间相互移动。

# 2 完整的训练过程
## 2.1 网络结构
### 2.1.1 定义网络的几种方法
1. 继承```nn.Module```类，定义一些**层**以及```.forward()```方法，返回值为**输出**
2. 使用```nn.Sequential()```，按顺序地定义每一层

### 2.1.2 torch.nn的特性
1. 可使用```net.parameters()```来查看模型所有的可学习参数，返回一个生成器
2. ```torch.nn```仅支持**一个batch**样本的输入(不支持单样本)，如果只有单个样本，需要手动添加维度

# 参考资料
1. [PyTorch中的contiguous - 栩风](https://zhuanlan.zhihu.com/p/64551412)