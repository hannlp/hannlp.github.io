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
这里引用[知乎-予以初始](https://www.zhihu.com/question/439243827/answer/1712516368)的回答，非常通俗易懂

**RNN**用于信息传输通路只有一条，并且该通路上的计算包含多次非线性激活操作。长记忆丢失是因为梯度消失，而梯度消失的主谋就是多层激活函数的嵌套，导致梯度反传时越乘越小（激活函数的导数<=1），乃至下溢出。所以后面的梯度传递不到前方，无法建立长时依赖。

**LSTM**引入了两条计算通道(**C**和**h**) 用于信息传输，其中**C**通道上的计算相对简单，较多的是矩阵的线性转换，没有太多的非线性激活操作。梯度反传时可以在**C**通道上平稳的传输到前方，从而建立长时依赖。所以**C**通道主要用于建立长时依赖，**h**通道用于建立短时依赖。

要说的是，LSTM的设计只是较RNN**缓解**了梯度消失问题，并没有完全解决。与Transformer的自注意力相比，LSTM的顺序输入的方式影响了模型的并行性，但符合人对序列的理解方式。

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
4. [混合前端的seq2seq模型部署](https://cloud.tencent.com/developer/article/1507554)