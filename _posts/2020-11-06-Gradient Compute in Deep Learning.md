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
> 注: 在本博客中，所有向量<span>$\bm{v}$</span>默认都为列向量

## 1.1 深度学习中几种常见的求导
在神经网络中，很常见的求导类型是一个**实值函数$f$**(如损失函数)对**一个向量$\bm{x}$**(如网络某一层的输出)或**一个矩阵$\bm{W}$**(如网络中的参数)进行求导，这些求导的实质其实就是多元函数求导，即求自变量关于函数值的梯度。

### 1.1.1 常数(实值函数)对向量的求导
$$\frac{\partial f}{\bm{x}},其中\bm{x}=\begin{pmatrix} x_1 \\ x_2 \\ ... \\x_n \end{pmatrix}.等价于\begin{pmatrix} \frac{\partial f}{x_1} \\ \frac{\partial f}{x_2} \\ ... \\\frac{\partial f}{x_n} \end{pmatrix}$$

### 1.1.2 常数(实值函数)对矩阵的求导
同1.1.1，即使用$f$依次对矩阵的每一个元素求导，结果仍然为一个矩阵。
# 2 深度神经网络(DNN)中反向传播的推导

## 2.1 变量定义

|    符号    |                          含义                          |
| :--------: | :----------------------------------------------------: |
|  $\bm{x}$  |          一个输入样本，维度为($N_0\times 1$)           |
| $W_{jk}^l$ | 第$l-1$层第$k$个神经元到第$l$层第$j$个神经元的连接权重 |
|  $b_j^l$   |               第$l$层第$j$个神经元的偏置               |
|  $z_j^l$   |             第$l$层第$j$个神经元的加权输入             |
| $\sigma^l$ |                   第$l$层的激活函数                    |
|  $a_j^l$   |               第$l$层第$j$个神经元的输出               |
|   $N_l$    |                   第$l$层神经元数量                    |

## 2.2 前向传播
为方便考虑数据的流动，我们首先仅考虑一个样本$\bm{x}$在DNN中的前向传播。

很容易发现，样本在每一层中流动的过程都是一样的，可以表示为:
$$
\begin{aligned}
\bm{z^l}=\bm{W^la^{l-1}+b^l}\\
\bm{a^l}=\sigma^l(\bm{z^l})     
\end{aligned}
$$

其中，$\bm{a^0}=\bm{x}$，$\bm{W^l}的维度为(N_{l}\times N_{l-1})$,$\bm{a^{l-1}}$的维度为$(N_{l-1}\times1)$,$\bm{b^l,z^l,a^l}$的维度均为$(N_l\times1)$，$\bm{a^L}$是DNN的输出

## 2.3 接下来要做的事情
通过前向传播，样本$\bm{x}$一层一层地通过DNN走到了输出层，并得到了$\bm{a^L}$。那么我们就能够通过$\bm{a^L}$和真实的标签$\bm{y}$得到这个样本的损失，即$C=Loss(\bm{a^L,y})$。

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
$b_j^l$会对$C$有什么影响呢？他只会作用在第$l$层的第$j$个节点上，作为一个小小的偏置。所以我们依然可以根据链式法则得到$\frac{\partial C}{\partial b_j^l}=\frac{\partial C}{\partial z_j^l}\frac{\partial z_j^l}{\partial b_j^l}$。其中，$z_j^l=\sum_{i=1}^{N_{l-1}}a_i^{l-1}W_{ji}^l+b_j^l$(与2.3.1中是一样的)

所以，$\frac{\partial z_j^l}{\partial b_j^l}=1$

所以，$\frac{\partial C}{\partial b_j^l}=\frac{\partial C}{\partial z_j^l}\times1=\frac{\partial C}{\partial z_j^l}$

## 2.4 (误差的)反向传播
通过上2.3节，可以看到如果想求$\bm{W}$和$\bm{b}$中每一个元素的梯度，都需要求$\frac{\partial C}{\partial z_j^l}$这一项。该怎么求呢？接下来就是反向传播的精髓了。

我小小的总结一下，前向传播是**训练样本**的前向传播，目的是使用$\bm{x}$通过DNN得到$\bm{a^L}$，从而计算$Loss(\bm{a^L,y})$。那么反向传播是**误差**的反向传播，即首先根据$Loss$得到最后一层，即第$L$层的误差，再反向计算每一层的误差。这里的误差，也就是我们刚刚要求的$\frac{\partial C}{\partial \bm{z^l}}$这一项。通过计算该项，我们便可以得到模型中所有参数的梯度，从而使用基于梯度的优化算法进行参数更新，这就是DNN完整的一轮迭代。

### 2.4.1 计算第$L$层的误差$\frac{\partial C}{\partial\bm{z^L}}$

同样的方法，我们还是先只考虑这一层中第$j$个神经元的误差$\frac{\partial C}{\partial z_j^L}$。

首先考虑$z_j^L$会对$C$产生什么影响。很简单的，$z_j^L$在被激活函数$\sigma^L$激活得到$a_j^L$然后作为计算$C$的一部分。

所以，又根据链式法则，得到$\frac{\partial C}{\partial z_j^L}=\frac{\partial C}{\partial a_j^L}\frac{\partial a_j^L}{\partial z_j^L}$
其中$a_j^L=\sigma^L(z_j^l)$,所以有$\frac{\partial a_j^L}{\partial z_j^L}=\sigma'^L(z_j^L)$
而$\frac{\partial C}{\partial a_j^L}$这一项需要根据具体的损失函数$Loss()$来计算。我们以平方损失为例：

$C=Loss(\bm{a^L,y})=\frac{1}{2}\Vert\bm{y-a^L}\Vert^2=\frac{1}{2}\sum_j(y_j-a_j^L)^2$

那么，我们便可以求得$\frac{\partial C}{\partial a_j^L}=\frac{\partial \frac{1}{2}[(y_1-a_1^L)^2+(y_2-a_2^L)^2+...+(y_{N_L}-a_{N_L}^L)^2]}{{\partial a_j^L}}=\frac{1}{2}\times2(y_j-a_j^L)\times-1=a_j^L-y_j$

可以看到$\frac{\partial C}{\partial a_j^L}和\frac{\partial a_j^L}{\partial z_j^L}$都是只与下标$j$有关的，所以我们可以直接将其扩展成向量形式，即$\frac{\partial C}{\partial\bm{z^L}}=(\bm{a^L-y})\bigodot\sigma'^L(\bm{z^L})$，其中$\bigodot$是两个向量的按元素乘法

从二次损失扩展到其他各种损失函数，即$\frac{\partial C}{\partial\bm{z^L}}=(\frac{\partial C}{\partial \bm{a^L}})\bigodot\sigma'^L(\bm{z^L})$

### 2.4.1 误差从第$l+1$层传播到第$l$层
为了计算误差在两层之间是怎么流动的，我们首先需要观察一下两层之间$\bm{z}$的关系，很简单，就是$\bm{z^{l+1}}=\bm{W^{l+1}}\sigma(\bm{z^l})+\bm{b^{l+1}}$

我们可以从上面的式子观察一下第$l$层的第$j$个神经元的$z_j^l$是怎么作用到下一层的。同样很简单，$z_j^l$会先经过一个激活函数$\sigma$得到$a_j^l$，再乘上不同的权重，作用在下一层的每一个神经元上。

所以，根据链式法则，有$\frac{\partial C}{\partial z_j^l}=\sum_k\frac{\partial C}{\partial z_k^{l+1}}\frac{\partial z_k^{l+1}}{\partial z_j^l}$

我们先看$\frac{\partial z_k^{l+1}}{\partial z_j^l}$这一项。$z_k^{l+1}$是怎么得到的呢？是上一层所有的$z_i^l$经过一个激活函数，再乘一个权重，最后加上一个偏置得到的，即$z_k^{l+1}=\sum_i^{N_l}\sigma^l(z_i^l)\times W_{ki}^{l+1}+b_k^{l+1}$

所以$\frac{\partial z_k^{l+1}}{\partial z_j^l}=\sigma'^l(z_j^l)\times W_{kj}^{l+1}$(仅当$i=j$时求和项不为0)

把他带回$\frac{\partial C}{\partial z_j^l}$，得到$\frac{\partial C}{\partial z_j^l}=\sum_k\frac{\partial C}{\partial z_k^{l+1}}\times W_{kj}^{l+1}\times\sigma'^l(z_j^l)$。

展开求和项，得到$\frac{\partial C}{\partial z_j^l}=[\frac{\partial C}{\partial z_1^{l+1}}\times W_{1j}^{l+1}+\frac{\partial C}{\partial z_2^{l+1}}\times W_{2j}^{l+1}+...+\frac{\partial C}{\partial z_{N_{l+1}}^{l+1}}\times W_{N_{l+1}j}^{l+1}]\times \sigma'^l(z_j^l)$

仔细观察其向量表示！

很惊讶的发现，上式$=[\bm{(W^{l+1})}^\mathrm{T}(\frac{\partial C}{\partial \bm{z^{l+1}}})]_j\times \sigma'^l(z_j^l)$

因为只由下标$j$决定，所以又可以得到一个完美漂亮的向量表达~

$\frac{\partial C}{\bm{\partial \bm{z^l}}}=\bm{(W^{l+1})}^\mathrm{T}(\frac{\partial C}{\partial \bm{z^{l+1}}})\bigodot\sigma'^l(\bm{z^l})$

# 参考文献
1.[Matrix Cookbook - Kaare Brandt Petersen, Michael Syskind Pedersen](https://cdn.jsdelivr.net/gh/hannlp/Books@1.01/Matrix%20Cookbook.pdf)