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
        \frac{e^{x_1}}{\sum_je^{x_j}}\\
        ...\\
        \frac{e^{x_i}}{\sum_je^{x_j}}\\
        ...\\
        \frac{e^{x_n}}{\sum_je^{x_j}}\\
    \end{bmatrix}
\end{aligned}
$$
若$\bm{y}=\mathrm{softmax}(\bm{x})$，那么对于任意$y_i$有以下特点：
1. $y_i\in[0,1]$，且$\sum_iy_i=1$，所以可以$y_i$当成属于类$i$的概率
2. 在计算任意一个$y_i$时，都会用到所有$x_i$

## 1.2 文档中的Softmax
在PyTorch或Tensorflow里，是这样描述Softmax的：
> take logits and produce probabilities



# 2 关于CrossEntropy

# 3 分类问题的梯度计算
