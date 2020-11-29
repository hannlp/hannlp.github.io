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
* [动手学深度学习-PyTorch](https://tangshusen.me/Dive-into-DL-PyTorch/#/) - 只需了解基础的线性代数、微分和概率，以及基础的Python编程，即可迅速入门PyTorch
* [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) - 提供了很多非常简洁的模板代码，也很适合学习使用

# 1 几点注意
## 1.1 内存中的tensor
由于python对于变量在内存中的特殊储存方式，基于python的PyTorch也会因此受到影响，具体有以下几种形式：
1. 像Numpy一样，对一个tensor使用索引操作(如new_tensor=tensor[1:])，索引出的结果与这个tensor**共享内存**(即修改一个，另一个也会跟着修改)
2. 用view()改变tensor的形状，返回的新tensor与源tensor**共享内存**(顾名思义，view()仅仅改变对该张量的观察角度，内部数据并未改变)。所以如果想返回一个**真正副本**，推荐使用tensor.clone.view()
3. 使用numpy()和from_numpy()将tensor与Numpy中的array相互转换时，产生的tensor和array**共享内存**。如果这个tensor需要一个新的内存，那么可以使用torch.tensor()，这将消耗更多的时间和空间。

## 1.2 inplace操作
* PyTorch操作inplace版本都有后缀_，代表就地修改，例如：
```python
y.add_(x)
y.copy_(x)
x.grad.data.zero_()
x.requires_grad_()
```

## 1.3 tensor在cpu或gpu
待更新
