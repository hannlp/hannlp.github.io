---
title: Full Matrix Method of mini-batch Back Propagation in DNN
date: 2020-11-30
tags:
- 深度学习
---

# 前言
在DNN的反向传播算法中，几乎所有教材都只使用**单个样本**(一个特征向量)进行相关公式的推导，而**多个样本**(也就是**mini-batch**，即多个特征向量组成的矩阵)反向传播的**全矩阵方法**对于理解“多样本”这一概念是非常重要的。所以花了一点时间推导了一下并记录在此，便于记忆，同时希望能对别人有所帮助。

# 0 几点说明
1. 该文章是我上一篇博客[Back Propagation and Gradient Calculation in Deep Learning](https://hannlp.github.io/2020-11-06-Back-Propagation-and-Gradient-Calculation-in-Deep-Learning/)(单样本反向传播)的后续版本，写作风格、符号表示与上篇类似，可以按顺序阅读
2. 只要是变量(向量或矩阵)关于**标量**的梯度，**其梯度的维度**一定与**其本身的维度**相同，这可以作为很多梯度推导的检验方法
3. 多样本在DNN的前向传播和反向传播中，**样本(列)与样本(列)**之间是毫无影响的。这一点**符合直觉**，也属于**矩阵的特性**，具体会在之后的推导过程中详细解释

# 1 变量定义

|    符号     |                               含义                               |
| :---------: | :--------------------------------------------------------------: |
|    $N_l$    |                        第$l$层神经元数量                         |
|  $\bm{X}$   |               $M$个输入样本，维度为($N_0\times M$)               |
|  $\bm{Y}$   |           这$M$个输入样本的标签，维度为($N_L\times M$)           |
| $W_{jk}^l$  | 第$l-1$层第$\bm{k}$个神经元到第$l$层第$\bm{j}$个神经元的连接权重 |
|   $b_j^l$   |                    第$l$层第$j$个神经元的偏置                    |
| $Z_{j,m}^l$ |         第$l$层第$j$个神经元对**第$m$个样本**的带权输入          |
| $\sigma^l$  |                        第$l$层的激活函数                         |
| $A_{j,m}^l$ |           第$l$层第$j$个神经元对**第$m$个样本**的输出            |


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

显而易见 (因为在$C$的展开式中，**仅此一项的导数不为0**)，$$\frac{\partial C}{\partial A_{j,m}^L}=\frac{1}{2m}\times2(Y_{j,m}-A_{j,m}^L)\times-1=\frac{1}{m}\cdot(A_{j,m}^L-Y_{j,m})$$


拓展到**矩阵表示**，即：
> $$\frac{\partial C}{\partial \bm{A^L}}=\frac{1}{m}\cdot(\bm{A^L}-\bm{Y})$$

第二步，需要求输出层误差$\frac{\partial C}{\partial \bm{Z^L}}$。由链式法则，$\frac{\partial C}{\partial \bm{Z^L}}=\frac{\partial C}{\partial \bm{A^L}}\cdot\frac{\partial \bm{A^L}}{\partial \bm{Z^L}}$.其中$\frac{\partial \bm{A^L}}{\partial \bm{Z^L}}$这一项等于$\sigma'^L(\bm{Z^L})$。因为$\bm{A^L}=\sigma^L(\bm{Z^L})$

所以，拓展到**任意损失函数**，输出层误差为：
> $$\frac{\partial C}{\partial \bm{Z^L}}=\frac{\partial C}{\partial \bm{A^L}}\odot\sigma'^L(\bm{Z^L})$$

# 5 反向传播误差
与上一篇文章相同，误差的反向传播，即已知$\frac{\partial C}{\partial \bm{Z^{l+1}}}$，希望求得$\frac{\partial C}{\partial \bm{Z^l}}$。我们还是先求矩阵中的一个元素的梯度$\frac{\partial C}{\partial Z_{j,m}^L}$。

在这之前，希望大家想起文章开头说过，**样本(列)与样本(列)之间是互不影响的**，怎么理解呢？也就是**每个样本(列)在每一层流动的过程中，永远都只在属于自己的那一列**。这也是为什么多样本反向传播误差时的公式可以和单样本的基本相似，只是多了一个下标$m$：

$$\frac{\partial C}{\partial Z_{j,m}^l}=\sum_k\frac{\partial C}{\partial Z_{k,m}^{l+1}}\frac{\partial Z_{k,m}^{l+1}}{\partial Z_{j,m}^l}$$  

其中，$Z_{k,m}^{l+1}=\sum_{i=1}^{N_l}\sigma^l(Z_{i,m}^l)\cdot W_{ki}^{l+1}+b_k^{l+1}$，有：

$$\frac{\partial Z_{k,m}^{l+1}}{\partial Z_{j,m}^l}=\sigma'^l(Z_{j,m}^l)\cdot W_{kj}^{l+1}$$

将其带回原式，

$$\begin{aligned}
    \frac{\partial C}{\partial Z_{j,m}^l}&=\sum_k\frac{\partial C}{\partial Z_{k,m}^{l+1}}\cdot W_{kj}^{l+1}\cdot\sigma'^l(Z_{j,m}^l)\\
    &=[\bm{(W^{l+1})}^\mathrm{T}(\frac{\partial C}{\partial \bm{Z^{l+1}}})]_{j,m}\cdot \sigma'^l(Z_{j,m}^l)
\end{aligned}
$$

拓展到**矩阵表示**，即：
> $$\frac{\partial C}{\bm{\partial \bm{Z^l}}}=\bm{(W^{l+1})}^\mathrm{T}(\frac{\partial C}{\partial \bm{Z^{l+1}}})\odot\sigma'^l(\bm{Z^l})$$

读过上一篇文章的朋友可能会发现，这个公式其实就是把上篇文章中的$\bm{z}$换成了$\bm{Z}$。没错，正因为**样本(列)与样本(列)之间互不影响**，误差的反向传播成为这篇文章中较为简单的一部分。

# 6 求$\bm{W^l,b^l}$的梯度
在考虑$\bm{W^l,b^l}$的时候总觉得有些奇怪，因为他们总是作用到所有样本上。不过不慌，我们还是先使用我们的法宝：只考虑矩阵或向量中一个元素的梯度$\frac{\partial C}{\partial W_{jk}^l}$和$\frac{\partial C}{\partial b_j^l}$。

## 6.1 求$\frac{\partial C}{\partial W_{jk}^l}$
先来与上一篇文章做一下对比：
>* 在**单样本前向传播**中，$W_{jk}^l$只会与第$l-1$层第$\bm{k}$个节点的输出$a_k^{l-1}$相乘，然后作为一部分汇聚到下一层，也就是第$l$层的第$\bm{j}$个节点上，这可以看作**一对一连接**。公式为：$$z_j^l=\sum_{i=1}^{N_{l-1}}a_i^{l-1}W_{ji}^l+b_j^l$$
>* 而在**多样本前向传播**中，$W_{jk}^l$会**分别**与第$l-1$层第$\bm{k}$个节点的$M$个输出$A_{k,m}^{l-1}$相乘，然后**分别**作为一部分汇聚到下一层，也就是第$l$层的第$\bm{j}$个节点上(共有$M$个)，这可以看作是**M个一对一连接**。公式为:$$Z_{j,m}^l=\sum_{i=1}^{N_{l-1}}A_{i,m}^{l-1}W_{ji}^l+b_j^l$$

所以，根据链式法则，我们要在$m$维度上累加：$$\frac{\partial C}{\partial W_{jk}^l}=\sum_m\frac{\partial C}{\partial Z_{j,m}^l}\frac{\partial Z_{j,m}^l}{\partial W_{jk}^l}$$

其中，$\frac{\partial Z_{j,m}^l}{\partial W_{jk}^l}=A_{k,m}^{l-1}$。代入原式得到$\frac{\partial C}{\partial W_{jk}^l}=\sum_m\frac{\partial C}{\partial Z_{j,m}^l}A_{k,m}^{l-1}$

拓展到**矩阵表示**，即：
> $$\frac{\partial C}{\partial \bm{W^l}}=\frac{\partial C}{\partial \bm{Z^l}}\cdot (\bm{A^{l-1})^\mathrm{T}}$$

使用维度计算验证一下：$(N_l\times M)\cdot(M \times N_{l-1})=(N_l\times N_{l-1})$

## 6.2 求$\frac{\partial C}{\partial b_j^l}$
由于$\bm{b^l}$在前向传播时，广播到了$M$个，所以与$\bm{W}$同理，在使用链式法则是同样要在$m$维度上累加：$$\frac{\partial C}{\partial b_j^l}=\sum_m\frac{\partial C}{\partial Z_{j,m}^l}\frac{\partial Z_{j,m}^l}{\partial b_j^l}$$

其中，$\frac{\partial Z_{j,m}^l}{\partial b_j^l}=1$。带入原式得到$\frac{\partial C}{\partial b_j^l}=\sum_m\frac{\partial C}{\partial Z_{j,m}^l}$

在扩展到**矩阵表示**时，我们需要引入一个全1的$M$维列向量：
> $$\frac{\partial C}{\partial \bm{b^l}}=\frac{\partial C}{\partial \bm{Z^l}}\cdot \mathrm{ones(M,1)}$$

使用维度计算验证一下: $(N_l\times M)\cdot(M,1)=(N_l\times 1)$

# 7 多样本反向传播全矩阵方法的最后公式
赏心悦目的公式们：
> $$\begin{aligned}
    \frac{\partial C}{\partial \bm{Z^L}}&=\frac{\partial C}{\partial \bm{A^L}}\odot\sigma'^L(\bm{Z^L})\\
    \frac{\partial C}{\bm{\partial \bm{Z^l}}}&=\bm{(W^{l+1})}^\mathrm{T}(\frac{\partial C}{\partial \bm{Z^{l+1}}})\odot\sigma'^l(\bm{Z^l})\\
    \frac{\partial C}{\partial \bm{W^l}}&=\frac{\partial C}{\partial \bm{Z^l}}\cdot (\bm{A^{l-1})^\mathrm{T}}\\
    \frac{\partial C}{\partial \bm{b^l}}&=\frac{\partial C}{\partial \bm{Z^l}}\cdot \mathrm{ones(M,1)}
\end{aligned}$$

# 8 与基于单样本累加的mini-batch更新方法的比较
Michael Nielsen在书中给出了计算mini-batch梯度的一种方法。

![](https://i.loli.net/2020/12/06/JFGIKhDXSU61fs7.png)

这个方法是在单样本反向传播基础上实现的，总体的思路是先**依次**进行**mini-batch个样本**的前向传播和误差计算，再依次传播误差到每一层中，将梯度**累积**起来，最后通过**平均**计算所有$\bm{W^l,b^l}$的梯度(即:$\frac{总梯度}{mini-batch}$)。而全矩阵方法是**并行化**的，也就是一次前向传播就可以同时完成**mini-batch个样本**的相应计算，反向传播误差时亦如此。并行化也在当今的算法中大受推崇，比如序列建模中，Transformer相对于(RNN家族)的一个巨大优势，就是可以并行计算。