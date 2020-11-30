---
title: mini-batch Back Propagation in DNN
date: 2020-11-30
tags:
- 基础知识
- 深度学习
---

# 前言
在DNN的反向传播算法中，几乎所有教材都只使用**单个样本**(一个特征向量)进行相关公式的推导，而**多个样本**(也就是**mini-batch**，即多个特征向量组成的矩阵)反向传播的**全矩阵方法**对于理解“多样本”这一概念是非常重要的。所以花了一点时间推导了一下并记录在此，便于记忆，同时希望能对别人有所帮助。

# 0 几点说明
1. 该文章是我上一篇博客[Back Propagation and Gradient Calculation in Deep Learning](https://hannlp.github.io/2020-11-06-Back-Propagation-and-Gradient-Calculation-in-Deep-Learning/)(单样本反向传播)的后续版本，写作风格、符号表示与上篇类似，请按顺序阅读
2. 请时刻记住，不论是标量，还是向量、矩阵，**其梯度的维度**一定与**其本身的维度**相同，这可以作为很多梯度推导的检验方法
3. 多样本在DNN的前向传播和反向传播中，**样本与样本**之间是毫无影响的。这一点**符合直觉**，也属于**矩阵的特性**，具体会在之后的推导过程中详细解释

# 1 变量定义

|    符号     |                               含义                               |
| :---------: | :--------------------------------------------------------------: |
|    $N_l$    |                        第$l$层神经元数量                         |
|  $\bm{X}$   |               $M$个输入样本，维度为($N_0\times M$)               |
|  $\bm{Y}$   |           这$M$个输入样本的标签，维度为($N_L\times M$)           |
| $W_{jk}^l$  | 第$l-1$层第$\bm{k}$个神经元到第$l$层第$\bm{j}$个神经元的连接权重 |
|   $b_j^l$   |                    第$l$层第$j$个神经元的偏置                    |
| $Z_{j,m}^l$ |           第$l$层第$j$个神经元对第$m$个样本的带权输入            |
| $\sigma^l$  |                        第$l$层的激活函数                         |
| $A_{j,m}^l$ |             第$l$层第$j$个神经元对第$m$个样本的输出              |


# 2 单样本到多样本的变化
变量定义已经给出，先说明一下**多样本**相对于**单样本**发生的变化

## 2.1 什么变了
1. 输入和标签变了，均由一个列向量变成一个矩阵($M$个列向量)：$$\bm{x}\in \bm{\mathrm{R^{N_0}}}\rightarrow\bm{X}\in \bm{\mathrm{R^{N_0\times M}}}，\bm{y}\in \bm{\mathrm{R^{N_L}}}\rightarrow\bm{Y}\in \bm{\mathrm{R^{N_L\times M}}}$$
2. 每一层的带权输入和输出变了，均由一个列向量变为一个矩阵($M$个列向量)：$$\bm{z^l}\in \bm{\mathrm{R^{N_l}}}\rightarrow\bm{Z^l}\in \bm{\mathrm{R^{N_l\times M}}}，\bm{a^l}\in \bm{\mathrm{R^{N_l}}}\rightarrow\bm{A^l}\in \bm{\mathrm{R^{N_l\times M}}}$$
3. 损失函数$Loss$变了，需要对所有样本的误差取平均了

## 2.2 什么没变
1. 不变的永远都是模型参数$\bm{W,b}$，与之前一样，每一层$l=1,2,...,L$都分别只有一个对应的$\bm{W^l}$和$\bm{b^l}$
2. 我们的目标依然不变，最终目的仍然是求得$\bm{W^l}$和$\bm{b^l}$的梯度

# 3 前向传播

$$
\begin{aligned}
&\bm{Z^l}=\bm{W^lA^{l-1}+\hat{b}^l}\\
&\bm{A^l}=\sigma^l(\bm{Z^l})     
\end{aligned}
$$

其中，$\bm{A^0}=\bm{X}$，$\bm{W^l}$的维度为$(N_{l}\times N_{l-1})$，$\bm{A^{l-1}}$的维度为$(N_{l-1}\times M)$

尤其注意$\bm{\hat{b}^l}$比$\bm{b^l}$多了个**帽子**，表示$\bm{b^l}$的**广播**(即直接将$\bm{b^l}$复制$M$个组成矩阵，维度与$\bm{Z^l,A^l}$相同，均为$(N_l\times M)$。这种广播是在两个维度不同的张量之间加减运算时产生的，在Numpy或PyTorch中是自动发生的)

$\bm{A^L}$是DNN的输出，维度为$(N_L\times M)$

# 4 计算损失并求输出层误差
同样以**平方损失**为例：

$$C=Loss(\bm{A^L,Y})=\frac{1}{m}\cdot\frac{1}{2}\Vert\bm{Y-A^L}\Vert^2=\frac{1}{m}\sum_m\frac{1}{2}\sum_j(Y_{j,m}-A_{j,m}^L)^2$$

第一步，需要求$\frac{\partial C}{\partial \bm{A^L}}$，我们同样先求矩阵的一个元素的梯度$\frac{\partial C}{\partial A_{j,m}^L}$

显而易见，$$\frac{\partial C}{\partial A_{j,m}^L}=\frac{1}{2m}\times2(Y_{j,m}-A_{j,m}^L)\times-1=\frac{1}{m}\cdot(A_{j,m}^L-Y_{j,m})$$
因为在$C$的展开式中，**仅此一项的导数不为0**(可以回顾前一篇文章的这个部分)。

拓展到**矩阵表示**，即：
> $$\frac{\partial C}{\partial \bm{A^L}}=\frac{1}{m}\cdot(\bm{A^L}-\bm{Y})$$

第二步，需要求输出层误差$\frac{\partial C}{\partial \bm{Z^L}}$。由链式法则，$\frac{\partial C}{\partial \bm{Z^L}}=\frac{\partial C}{\partial \bm{A^L}}\cdot\frac{\partial \bm{A^L}}{\partial \bm{Z^L}}$.其中$\frac{\partial \bm{A^L}}{\partial \bm{Z^L}}$这一项等于$\sigma'^L(\bm{Z^L})$。因为$\bm{A^L}=\sigma^L(\bm{Z^L})$

所以，拓展到**任意损失函数**，输出层误差为：
> $$\frac{\partial C}{\partial \bm{Z^L}}=\frac{\partial C}{\partial \bm{A^L}}\odot\sigma'^L(\bm{Z^L})$$

# 5 反向传播误差
与上一篇文章相同，误差的反向传播，即已知$\frac{\partial C}{\partial \bm{Z^{l+1}}}$，希望求得$\frac{\partial C}{\partial \bm{Z^l}}$。我们还是先求矩阵中的一个元素的梯度$\frac{\partial C}{\partial Z_{j,m}^L}$。

在这之前，希望大家想起文章开头说过，**样本与样本之间是互不影响的**，怎么理解呢？也就是**每个样本在每一层流动的过程中，永远都只在属于自己的那一列**。这也是为什么多样本反向传播误差时的公式可以和单样本的基本相似，只是多了一个下标$m$
$$\begin{aligned}
    &\frac{\partial C}{\partial Z_{j,m}^l}=\sum_k\frac{\partial C}{\partial Z_{k,m}^{l+1}}\frac{\partial Z_{k,m}^{l+1}}{\partial Z_{j,m}^l}\\
    其中，&Z_{k,m}^{l+1}=\sum_{i=1}^{N_l}\sigma^l(Z_{i,m}^l)\times W_{ki}^{l+1}+b_k^{l+1}\\
    所以，&\frac{\partial Z_{k,m}^{l+1}}{\partial Z_{j,m}^l}=\sigma'^l(Z_{j,m}^l)\times W_{kj}^{l+1}\\
    所以，\frac{\partial C}{\partial Z_{j,m}^l}&=\sum_k\frac{\partial C}{\partial Z_{k,m}^{l+1}}\times W_{kj}^{l+1}\times\sigma'^l(Z_{j,m}^l)\\
    &=[\bm{(W^{l+1})}^\mathrm{T}(\frac{\partial C}{\partial \bm{Z^{l+1}}})]_{j,m}\times \sigma'^l(Z_{j,m}^l)
\end{aligned}
$$

拓展到**矩阵表示**，即：
> $$\frac{\partial C}{\bm{\partial \bm{Z^l}}}=\bm{(W^{l+1})}^\mathrm{T}(\frac{\partial C}{\partial \bm{Z^{l+1}}})\odot\sigma'^l(\bm{Z^l})$$
> 
读过上一篇文章的朋友可能会发现，这个公式其实就是把上篇文章中的$\bm{z}$换成了$\bm{Z}$。没错，正因为**样本与样本之间互不影响**，误差的反向传播成为这篇文章中较为简单的一部分。

# 6 计算$\bm{W^l,b^l}$的梯度

