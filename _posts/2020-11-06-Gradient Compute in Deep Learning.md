---
title: Gradient Compute in Deep Learning
date: 2019-04-16 19:32:34
tags:
- 基础知识
- 深度学习
---

# 前言
- *这篇博客的初心* : 最近读的论文又用到LSTM了，发现对这些深度学习模型我还是只了解皮毛，不了解其底层原理(如参数的更新)，而我从接触深度学习开始就对反向传播充满了好奇，感觉这是个很难理解的事情。所以建立这篇博客慢慢从矩阵求导开始，慢慢推导所有深度学习模型的底层原理，从而加深自己的理解。
- *这篇博客内容* : 包括**部分深度学习所需数学知识**，以及**各种深度学习模型(DNN,RNN等)的原理推导**。

# 1 数学知识
> 注: 在本博客中，所有向量<span>$\bm{x}$</span>默认都为列向量

## 1.1 深度学习中几种常见的求导
在神经网络中，很常见的求导类型是一个**实值函数$f$**(如损失函数)对**一个向量$\bm{x}$**(如网络某一层的输出)或**一个矩阵$\bm{W}$**(如网络中的参数)进行求导，这些求导的实质其实就是多元函数求导，即求自变量关于函数值的梯度。

### 1.1.1 常数(实值函数)对向量的求导
$\frac{\partial f}{\bm{x}},其中\bm{x}=\begin{pmatrix} x_1 \\ x_2 \\ ... \\x_n \end{pmatrix}.等价于\begin{pmatrix} \frac{\partial f}{x_1} \\ \frac{\partial f}{x_2} \\ ... \\\frac{\partial f}{x_n} \end{pmatrix}$

### 1.1.2 常数(实值函数)对矩阵的求导
同1.1.1，即使用$f$依次对矩阵的每一个元素求导，结果仍然为一个矩阵。
# 2 深度神经网络(DNN)中反向传播的推导

## 2.1 变量定义
| 符号            | 含义                                                   |
| --------------- | ------------------------------------------------------ |
| $X_{m\times n}$ | 输入的数据，其中$m$为样本数，$n$为样本的维度           |
| $\bm{x}$        | 一个输入样本(维度为$n$)                                |
| $W_{jk}^l$      | 第$l-1$层第$k$个神经元到第$l$层第$j$个神经元的连接权重 |
| $b_j^l$         | 第$l$层第$j$个神经元的偏置                             |
| $z_j^l$         | 第$l$层第$j$个神经元的加权输入                         |
| $\sigma^l$      | 第$l$层的激活函数                                      |
| $a_j^l$         | 第$l$层第$j$个神经元的输出                             |
| $N_l$           | 第$l$层神经元数量                                      |

## 2.2 前向传播
为方便考虑数据的流动，我们首先仅考虑一个样本$\bm{x}$在DNN中的前向传播。

### 2.2.1 第一层(输入层)
$\bm{a^1}=\sigma^1(W^1\bm{x}+b^1),shapeof(\bm{a^1})=(N_1\times1)$

### 2.2.2 第$l$层(中间层)
$\bm{a^l}=\sigma^l(W^l\bm{a^{l-1}}+b^l),shapeof(\bm{a^l})=(N_l\times1)$

### 2.2.3 第$L$层(输出层)
$\bm{a^L}=\sigma^L(W^L\bm{a^{L-1}}+b^L),shapeof(\bm{a^L})=(N_L\times1)$

## 2.3 接下来要做的事情
通过前向传播，样本$\bm{x}$一层一层地通过DNN走到了输出层，并得到了$\bm{a^L}$。那么我们就能够通过$\bm{a^L}$和真实的标签$\bm{y}$得到这个样本的损失，即$C=Loss(\bm{a^L},\bm{y})$。

我们希望根据这个损失$C$来不断的更新DNN的参数$[\bm{W^1,W^2,...,W^L}及\bm{b^1,b^2,...,b^l}]$。为了使用基于梯度的优化算法，很显然需要计算所有参数的梯度，也就是$[\frac{\partial C}{\partial\bm{W^1}},\frac{\partial C}{\partial\bm{W^2}},...,\frac{\partial C}{\partial\bm{W^L}}]$及$[\frac{\partial C}{\partial\bm{b^1}},\frac{\partial C}{\partial\bm{b^2}},...,\frac{\partial C}{\partial\bm{b^L}}]$。

由1.1.1和1.1.2可知，对向量或矩阵求梯度，就是对其中的每一个元素分别求梯度，所以我们开始吧。

### 2.3.1 对$\bm{W^l}$求导
我们只考虑第$l$层参数矩阵中的一个元素$W_{jk}^l$的梯度，来看看有什么规律。
$W_{jk}^l$会对$C$有什么影响呢？他只会与$l-1$层第$k$个节点的输出$a_k^{l-1}$相乘，然后作为一部分汇聚到下一层，也就是第$l$层的第$j$个节点上。如图所示:

所以根据链式法则，有$\frac{\partial C}{\partial W_{jk}^l}=\frac{\partial C}{\partial z_j^l}\frac{\partial z_j^l}{\partial W_{jk}^l}$。其中，$z_j^l=\sum_{i=1}^{N_{l-1}}a_i^{l-1}W_{ji}^l+b_j^l$

所以，$\frac{\partial z_j^l}{\partial W_{jk}^l}=a_k^{l-1}$(仅当$i=k$时，求和项不为0)

所以，$\frac{\partial C}{\partial W_{jk}^l}=\frac{\partial C}{\partial z_j^l}a_k^{l-1}$

### 2.3.2 对$\bm{b^l}$求导
同样，我们只考虑$b_j^l$的梯度。


那么，$\frac{\partial C}{\partial z_j^l}$这一项该怎么求呢？



# 参考文献
1.[Matrix Cookbook - Kaare Brandt Petersen, Michael Syskind Pedersen](https://cdn.jsdelivr.net/gh/hannlp/Books@1.01/Matrix%20Cookbook.pdf)


