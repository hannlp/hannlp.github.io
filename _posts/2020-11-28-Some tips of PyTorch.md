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
* [动手学深度学习-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch) - 只需了解基础的线性代数、微分和概率，以及基础的Python编程，即可迅速入门PyTorch
* [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) - 提供了很多非常简洁的模板代码，也很适合学习使用

# 1 PyTorch与python的内存
