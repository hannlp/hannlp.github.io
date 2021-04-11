---
title: Notes of Torchtext a NLP processing tool
date: 2021-04-11
tags:
- 自然语言处理
---

# 前言

# 0 推荐资源
1. [TORCHTEXT DOCUMENTATION (0.8.1)](https://pytorch.org/text/0.8.1/) (官方文档，目前已更新到[0.9.0](https://pytorch.org/text/stable/index.html))
2. [pytorch/text](https://github.com/pytorch/text#data) (官方github仓库，其**readme**是一个非常简洁的使用指南)
3. [How to use TorchText for neural machine translation, plus hack to make it 5x faster](https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95#8a90) (一个优质的使用torchtext预处理机器翻译数据的教程)

> **版本提示：** 在0.9.0版本中，之前版本的很多重要模块如```torchtext.data```、```torchtext.datasets```等已经移动到```torchtext.legacy```中了，导入时需要注意  

# 1 使用记录