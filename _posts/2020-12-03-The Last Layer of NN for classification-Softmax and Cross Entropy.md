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
    \mathrm{Softmax}(\bm{x})=\begin{bmatrix}
        ...\\
        \frac{e^{x_i}}{\sum_je^{x_j}}\\
        c\\
        d\\
    \end{bmatrix}
\end{aligned}
$$
## 1.2 Softmax

# 2 关于CrossEntropy

# 3 分类问题的梯度计算
