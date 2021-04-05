---
title: Some Pytorch practices of LSTM
date: 2021-04-05
tags:
- 深度学习
---

# 前言
在复现Bahdana的注意力机制NMT模型时(PyTorch)，第一次使用LSTM作为编码器和解码器的基本结构，发现fairseq的源码中编码器使用LSTM，而解码器使用的却是LSTMCell，遂产生好奇。在深入研究之后才明白其原因，将经验心得记录在此。

# 1 LSTM速览
## 1.1 LSTM流程图
![](https://i.loli.net/2021/04/05/pCeGQALRIy2NVoc.png)

## 1.2 LSTM的关键
## 1.3 与RNN的对比

# 2 多层LSTM
![](https://i.loli.net/2021/04/05/scw8fu5I27DQjSN.png)

# 3 PyTorch中的LSTM
## 3.1 LSTM

## 3.2 LSTMCell

# 4 PyTorch实践：Encoder-Decoder模型
## 4.1 用LSTM写Encoder
## 4.2 用LSTMCell写带attention的Decoder

# 参考资料
1. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
2. [Fully understand LSTM network and input, output, hidden_size and other parameters](https://programmersought.com/article/91264364976/)
3. [LSTM细节分析理解（pytorch版）](https://zhuanlan.zhihu.com/p/79064602)