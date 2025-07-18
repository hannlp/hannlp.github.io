---
title: 关于Softmax和CrossEntropyLoss
date: 2020-12-03
tags:
- 深度学习
---

# 前言
本文对NLP中常用的**Softmax+CrossEntropyLoss组合**的符号梯度进行推导，证明其巧妙之处。

# 1 关于Softmax
## 1.1 Softmax的形式

$$若\bm{x}=\begin{bmatrix}
        x_1\\
        ...\\
        x_i\\
        ...\\
        x_n\\
    \end{bmatrix},那么\mathrm{Softmax}(\bm{x})=\begin{bmatrix}
        \frac{e^{x_1}}{\sum_ke^{x_k}}\\
        ...\\
        \frac{e^{x_i}}{\sum_ke^{x_k}}\\
        ...\\
        \frac{e^{x_n}}{\sum_ke^{x_k}}\\
    \end{bmatrix}$$

若$\bm{y}=\mathrm{Softmax}(\bm{x})$，那么对于任意$y_i$有以下特点：
1. $y_i\in(0,1)$，且$\sum_iy_i=1$，所以可以$y_i$当成属于类$i$的概率
2. 在计算任意一个$y_i$时，都会用到所有$x_i$
3. 在计算任意一个$y_i$时，都会以$e$为底数，我们知道$e^x$会随着$x$的增大而急剧增大，这就会产生一种“大的更大，小的更小”的**马太效应**

## 1.2 一些其他细节
1. **为什么叫这个名字？** 
其实$\mathrm{Softmax}$就是$\mathrm{soft}$版本的$\mathrm{max}$。我们平时所说的$\mathrm{max}$，就是从**多个值中选出一个最大的**，这其实是$\mathrm{Hardmax}$。而$\mathrm{Softmax}$是**分别给这些值一个相应的概率**，另外由于其有马太效应，数值相差越大，概率相差也越大。如果给其前面加一个$\mathrm{log}$，那么就是$\mathrm{max}$的一个可微的近似
2. 关于$\mathrm{Softmax}$其实还有很多细节，比如数值稳定性问题，本文就不一一展开讲了，可以参考[Softmax vs. Softmax-Loss: Numerical Stability](https://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/) 这篇文章，是一篇不错的延伸

# 2 关于CrossEntropy Loss
## 2.1 CrossEntropy
给定两个概率分布$p,q$，其交叉熵为：

$$H(p,q)=-\sum_xp(x)\mathrm{log}q(x)$$

它刻画了两个概率分布之间的距离。其中$p$代表正确分布，$q$代表的是预测分布。交叉熵越小，两个概率的分布越接近

## 2.2 CrossEntropy Loss
在分类问题中，提出了交叉熵损失。形式如同：

$$C=-\sum_iy_i\mathrm{log}\hat{y_i}$$

其中，$y_i$为真实标签，$\hat{y_i}$为预测结果中对应的分布。在本篇文章中，$\hat{y_i}$就对应了网络最后一层第$i$个位置的输出$a_i$，也就是$\frac{e^{z_i}}{\sum_k^N e^{z_k}}$。

另外，当类别数仅为$2$时，$\mathrm{CrossEntropy Loss}$就变为：

$$\begin{aligned}
    C&=-\sum_{i=0}^1y_i\mathrm{log}\hat{y_i}\\
    &=-[y_0\mathrm{log}\hat{y_0}+y_1\mathrm{log}\hat{y_1}]\\
    &=-[y\mathrm{log}\hat{y}+(1-y)\mathrm{log}(1-\hat{y})]
\end{aligned}$$

注：这里$y_1=1-y_0,\hat{y_1}=1-\hat{y_0}$，且省略下标

# 3 分类问题的梯度计算
## 3.1 变量定义
我们设有一个$L$层的神经网络。$\mathrm{Softmax}$函数只作用在最后一层，所以只需要考虑第$L$层即可(注：本篇文章中直接省略**表示层数的上标$L$**)：

|   符号   |                含义                 |      维度      |
| :------: | :---------------------------------: | :------------: |
|   $N$    |     第$L$层(最后一层)神经元数量     |      标量      |
| $\bm{z}$ |     第$L$层(最后一层)的带权输入     | $(N\times 1)$  |
| $\bm{a}$ |       第$L$层(最后一层)的输出       | $(N \times 1)$ |
| $\bm{y}$ | 类别标签，是一个$one$-$hot$类型向量 | $(N\times 1)$  |

## 3.2 各变量之间的关系
使用交叉熵损失函数，有：

$$\begin{aligned}
\bm{a}&=\mathrm{Softmax}(\bm{z})\\
    C&=Loss(\bm{a,y})=-\sum_i^Ny_i\cdot \mathrm{log}a_i
\end{aligned}$$

其中，$a_i=\frac{e^{z_i}}{\sum_k^N e^{z_k}}$。而$\bm{y}$的形式如同：$\begin{bmatrix}0&0&...&1&...&0\end{bmatrix}^\mathrm{T}$，即$y_i$仅在正确的类别处为1，其余位置处均为0。

## 3.3 求$\frac{\partial C}{\partial \bm{z}}$
要想反向传播梯度，首先需要先计算最后一层的误差$\frac{\partial C}{\partial \bm{z}}$。

遵循从单个到整体的求梯度原则，我们仍然只计算$\frac{\partial C}{\partial z_i}$。因为$z_i$会作用到每一个$a_j$当中，所以根据链式法则，有$$\frac{\partial C}{\partial z_i}=\sum_j^N\frac{\partial C}{\partial a_j}\cdot\frac{\partial a_j}{\partial z_i}$$

### 3.3.1 计算$\frac{\partial a_j}{\partial z_i}$

$$\begin{aligned}
    \frac{\partial a_j}{\partial z_i}&=\frac{\partial \frac{e^{z_j}}{\sum_k^N e^{z_k}}}{\partial z_i}\\
    &=\frac{\frac{\partial e^{z_j}}{\partial z_i}\cdot\sum_k^N e^{z_k}-\frac{\partial \sum_k^N e^{z_k}}{\partial z_i}\cdot e^{z_j}}{(\sum_k^N e^{z_k})^2}\ \qquad(1)
\end{aligned}$$

**1.当$i=j$时，有：**

$$\begin{aligned}
    (1)式&=\frac{e^{z_{i(j)}}\cdot\sum_k^N e^{z_k}-e^{z_{i(j)}}\cdot e^{z_j}}{(\sum_k^N e^{z_k})^2}\\
    &=a_{i(j)}-a_{i(j)}\cdot a_j\quad(\mathrm{Softmax}定义)
\end{aligned}
$$

> 注：这里的**下标 $_{i(j)}$**，意为在这时不论取$i$或取$j$都是一样的。下文同理

**2.当$i\not ={}j$时，有：**
 
$$\begin{aligned}
    (1)式&=\frac{0-e^{z_i}\cdot e^{z_j}}{(\sum_k^N e^{z_k})^2}\\
    &=-a_i\cdot a_j
\end{aligned}
$$

所以，

$$\frac{\partial a_j}{\partial z_i}=\left\{\begin{aligned}
    a_i-a_i\cdot a_j\qquad(i=j)\\
    -a_i\cdot a_j\qquad(i\not ={j})
\end{aligned} \right.$$

### 3.3.2 计算$\frac{\partial C}{\partial a_j}$
因为$\bm{y}$为$one$-$hot$向量，假设仅$y_k=1$，那么：

$$\begin{aligned}
    C&=-\sum_i^N y_i\mathrm{log}a_i=-y_k\mathrm{log}a_k\\
    \frac{\partial C}{\partial a_j}&=\left\{\begin{aligned}
    0\qquad(j\not ={k})\\
    -\frac{y_{j(k)}}{a_{j(k)}}\quad(j=k)
\end{aligned} \right.\\
    &=-\frac{y_j}{a_j}
\end{aligned}
$$

注：因为当$j\not ={k}$时$y_j=0$，所以可以很直接的将两种情况合并

### 3.3.3 将$\frac{\partial a_j}{\partial z_i}$和$\frac{\partial C}{\partial a_j}$带入$\frac{\partial C}{\partial z_i}$

$$\begin{aligned}
    \frac{\partial C}{\partial z_i}&=\sum_j^N\frac{\partial C}{\partial a_j}\cdot\frac{\partial a_j}{\partial z_i}\\
    &=-\sum_{j=1}^N\frac{y_j}{a_j}\cdot[a_i\cdot1\{i=j\}-a_ia_j]\qquad(1)\\
    &=-\frac{y_{i(j)}}{a_{i(j)}}(a_i-a_ia_{i(j)})+\sum_{j=1,j\not ={i}}^N\frac{y_j}{a_j}(a_ia_j)\qquad(2)\\
    &=-y_i+y_ia_i+\sum_{j=1,j\not ={i}}^Ny_ja_i\\
    &=-y_i+a_i(y_i+\sum_{j=1,j\not ={i}}^Ny_j)\qquad(3)\\
    &=a_i-y_i
\end{aligned}$$

注：
1. 在公式$(1)$中，1{..}为**示性函数**，大括号内表达式为真时函数值为$1$，否则为$0$
2. 在公式$(2)$中，其实是把公式$(1)$的求和项分成了两个部分，左半部分是$i=j$时的情况，所以这里加上了下标 $_{i(j)}$，代表可以任意替换，而右半部分是$i\not ={j}$的情况，就必须严格遵守原始下标
3. 在公式$(3)$中，括号中的表达式恒等于$1$(因为$\bm{y}$为$one$-$hot$向量)

因为$\frac{\partial C}{\partial z_i}$只与下标$i$有关，所以可以扩展到向量形式，这里我再顺便加上层数$L$:
> $$\frac{\partial C}{\partial \bm{z^L}}=\bm{a^L}-\bm{y}$$

## 3.4 分析与对比
这个组合的梯度意味着，如果我的分类网络中采用$\mathrm{Softmax+CrossEntropy Loss}$，在计算最后一层误差的时候，我只需要**记录最后一层的输出**，然后再在**正确的类别**的那个位置**减去1**就可以了！

再对比一下**回归问题**，若采用$\mathrm{MSE}$作为损失函数，使用除$\mathrm{Softmax}$外的其他激活函数$\sigma^L$作为最后一层的激活函数的话，很容易得到$\frac{\partial C}{\partial\bm{z^L}}=(\bm{a^L-y})\odot\sigma'^L(\bm{z^L})$，惊讶的发现他们竟如此的一致！

# 4 PyTorch中的CrossEntropyLoss
## 4.1 原理速览
具体用法见[官方文档](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=nn%20crossentropyloss#torch.nn.CrossEntropyLoss)，下面简单介绍原理。在PyTorch中，```nn.CrossEntropyLoss```的作用是把```softmax```、```log```和```nn.NLLLoss```合成一步，具体为：  
1. **softmax:** 在某一维度进行Softmax归一化，使这一维度均在0-1之间，且和为1
2. **log:** 对上面结果取log，由于定义域在(0,1)，取log后的值域就在(-inf,0)
3. **NLLLoss:** 按照类别标签，在与步骤1不同的另一维度上，仅抽取标签处的值，去掉负号并按```reduction```的方法降维成标量

在这里推荐[参考资料4](https://blog.csdn.net/qq_22210253/article/details/85229988)，里面有简单详细的例子，看一遍就会用了

## 4.2 一个实际的例子
在我复现Transformer时，需要定义```criterion```,由于机器翻译中译文的生成需要在词表中挑选词语，属于分类问题。所以直接选用```nn.CrossEntropyLoss```作为准则  
```python
criterion = nn.CrossEntropyLoss(ignore_index=args['tgt_pdx'], reduction='sum')
```
以下是```loss```的计算。其中```out```的维度为 **(batch_size, tgt_len, n_tgt_words)** ，```tgt_tokens```的维度为 **(batch_size, tgt_len)**。其中，```out```是模型最后一个DecoderLayer的输出，使用Linear层将模型维度映射到了目标语言词表维度  
```python
loss = self.criterion(out.reshape(-1, out.size(-1)), tgt_tokens.contiguous().view(-1))
```
这里做了个维度变换，将整个batch的句子，直接展开成所有词。```out```的维度变成了 **(n_total_words, n_tgt_words)** ，```tgt_tokens```的维度变成了 **(n_total_words)**，符合官方文档中的输入形式。最后通过```reduction='sum'```，将结果求和得到标量```loss```，用于autograd自动计算梯度

# 5 参考资料
1. [你 真的 懂 Softmax 吗？](https://zhuanlan.zhihu.com/p/90771255)
2. [softmax激活+crossEntropy损失求导公式推导](https://fengzhe.blog.csdn.net/article/details/99707296)
3. [深度学习之反向传播算法(2)——全连接神经网络的BP算法推导](https://zhuanlan.zhihu.com/p/61531989)
4. [Pytorch详解NLLLoss和CrossEntropyLoss](https://blog.csdn.net/qq_22210253/article/details/85229988)