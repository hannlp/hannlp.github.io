---
title: The Last Layer of NN for classification-Softmax and Cross Entropy
date: 2020-12-03
tags:
- 深度学习
---

# 前言
传统机器学习中两大经典任务就是**回归**与**分类**。分类在深度学习中也很常见，令我印象最深的是图像分类。当然，在NLP中，分类也无处不在。从RNN与其变体，到Transformer，Bert等预训练模型，只要涉及到在**词表**中**挑选单词**，就可以使用分类任务的思路来解决。在深度学习模型中，区分回归还是分类，往往只需要改变**最后一层的激活函数**以及**损失函数**。

# 1 关于Softmax
## 1.1 Softmax的形式
$$\begin{aligned}
    若\bm{x}=\begin{bmatrix}
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
    \end{bmatrix}
\end{aligned}
$$
若$\bm{y}=\mathrm{softmax}(\bm{x})$，那么对于任意$y_i$有以下特点：
1. $y_i\in[0,1]$，且$\sum_iy_i=1$，所以可以$y_i$当成属于类$i$的概率
2. 在计算任意一个$y_i$时，都会用到所有$x_i$

## 1.2 


# 2 关于CrossEntropy

# 3 分类问题的梯度计算
## 3.1 变量定义
我们设有一个$L$层的神经网络。$\mathrm{Softmax}$函数只作用在最后一层，所以只需要考虑第$L$层即可(注：本篇文章中直接省略**表示层数的上标$L$**)：

|   符号   |                含义                 |      维度      |
| :------: | :---------------------------------: | :------------: |
|   $N$    |     第$L$层(最后一层)神经元数量     |                |
| $\bm{z}$ |     第$L$层(最后一层)的带权输入     | $(N\times 1)$  |
| $\bm{a}$ |       第$L$层(最后一层)的输出       | $(N \times 1)$ |
| $\bm{y}$ | 类别标签，是一个$one$-$hot$类型向量 | $(N\times 1)$  |

## 3.2 各变量之间的关系
使用交叉熵损失函数，有$$C=Loss(\bm{a,y})=-\sum_i^Ny_i\cdot \mathrm{log}a_i$$
其中，$a_i=\frac{e^{z_i}}{\sum_k^N e^{z_k}}$。而$\bm{y}$的形式如同：$\begin{bmatrix}0&0&...&1&...&0\end{bmatrix}^\mathrm{T}$，即$y_i$仅在正确的类别处为1，其余位置处均为0。

## 3.3 求$\frac{\partial C}{\partial \bm{z}}$
要想反向传播梯度，首先需要先计算最后一层的误差$\frac{\partial C}{\partial \bm{z}}$。

遵循从单个到整体的求梯度原则，我们仍然只计算$\frac{\partial C}{\partial z_i}$。因为$z_i$会作用到每一个$a_j$当中，所以根据链式法则，有$$\frac{\partial C}{\partial z_i}=\sum_j^N\frac{\partial C}{\partial a_j}\cdot\frac{\partial a_j}{\partial z_i}$$

**我们先计算$\frac{\partial a_j}{\partial z_i}$这一项：**

$$\begin{aligned}
    \frac{\partial a_j}{\partial z_i}&=\frac{\partial \frac{e^{z_j}}{\sum_k^N e^{z_k}}}{\partial z_i}\\
    &=\frac{\frac{\partial e^{z_j}}{\partial z_i}\cdot\sum_k^N e^{z_k}-\frac{\partial \sum_k^N e^{z_k}}{\partial z_i}\cdot e^{z_j}}{(\sum_k^N e^{z_k})^2}\ (除法求导法则)\qquad(1)
\end{aligned}$$

1) 当$i=j$时，有：

$$\begin{aligned}
    式(1)&=\frac{e^z_{i(j)}\cdot\sum_k^N e^{z_k}-e^z_{i(j)}\cdot e^{z_j}}{(\sum_k^N e^{z_k})^2}\\
    &=a_{i(j)}-a_{i(j)}\cdot a_j
\end{aligned}
$$

(这里有下标$i(j)$，意为在这时不论取$i$或取$j$都是一样的)

2) 当$i\not ={}j$时，有：
 
$$\begin{aligned}
    式(1)&=\frac{0-e^{z_i}\cdot e^{z_j}}{(\sum_k^N e^{z_k})^2}\\
    &=-a_i\cdot a_j
\end{aligned}
$$

所以，

$$\frac{\partial a_j}{\partial z_i}=\left\{\begin{aligned}
    a_i-a_i\cdot a_j\qquad(i=j)\\
    -a_i\cdot a_j\qquad(i\not ={j})
\end{aligned} \right.$$

**再计算$\frac{\partial C}{\partial a_j}$这一项：**
因为$\bm{y}$为$one$-$hot$向量，假设仅$y_k=1$，那么：

$$
$$