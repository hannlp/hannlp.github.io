---
title: Backward Propogation Through Time (BPTT) in RNN
date: 2020-12-14
tags:
- 深度学习
---

# 前言
* _一点感悟：_ 前几天简单看了下王者荣耀觉悟AI的论文，发现除了强化学习以外，也用到了熟悉的**LSTM**。之后我又想起了知乎上的一个问题：“Transformer会彻底取代RNN吗？”。我想，在觉悟AI这类**严格依赖于时间**(比如：每读一帧，就要立即进行相应的决策) 的情境中，就根本没法用Transformer这类基于self-attention的模型。因为self-attention的独特性使其**必须在一开始就知道所有时间位置的信息**，Transformer在NLP上的成功，我觉得还是因为**自然语言**并不算是严格依赖于时间的。因为我们在**数据中**看到的句子都是**完整的一句话**，这就方便了self-attention直接对每个位置进行建模。
* 所以，**Transformer是不可能彻底取代RNN的**。当然这只是我的一点思考，还有其他重要的原因：比如Transformer、Bert这种基于self-attention结构的预训练模型都需要海量的训练数据。在数据量不足的情景，只会带来巨大的偏差，但相同的数据在RNN甚至LSTM上已经可以达到足够好的效果。
* 所以，RNN及其变种是永恒的经典，有必要认真学习。遂推导了一下RNN的反向传播算法(BPTT)，记录在此。

# 1 RNN模型结构及符号定义
## 1.1 模型结构
假设有一个时间序列$t=1,2,...,L$，在每一时刻$t$我们有：

$$\begin{aligned}
\bm{z^{(t)}}&=\bm{Ux^{(t)}}+\bm{Wh^{(t-1)}}+\bm{b}\\
    \bm{h^{(t)}}&=f(\bm{z^{(t)}})\\
    \bm{s^{(t)}}&=\bm{Vh^{(t)}}+\bm{c}\\
    \bm{y^{(t)}}&=g(\bm{s^{(t)}})
\end{aligned}$$

这就是RNN的结构。可以看到，每一时刻$t$的隐含状态$\bm{h^{(t)}}$都是由**当前时刻的输入$\bm{x^{(t)}}$** 和**上一个时刻的隐含状态$\bm{h^{(t-1)}}$** 共同得到的。下面是详细的符号定义：

## 1.2 符号定义

|      符号      |              含义              |      维度      |
| :------------: | :----------------------------: | :------------: |
| $\bm{x^{(t)}}$ |        第$t$时刻的输入         | $(K\times 1)$  |
| $\bm{z^{(t)}}$ |    第$t$时刻隐层的带权输入     | $(N\times 1)$  |
| $\bm{h^{(t)}}$ |      第$t$时刻的隐含状态       | $(N\times 1)$  |
| $\bm{s^{(t)}}$ |   第$t$时刻输出层的带权输入    | $(M \times 1)$ |
| $\bm{y^{(t)}}$ |        第$t$时刻的输出         | $(M\times 1)$  |
|   $E^{(t)}$    |        第$t$时刻的损失         |      标量      |
|    $\bm{U}$    | 隐层对输入的参数，整个模型共享 | $(N\times K)$  |
|    $\bm{W}$    | 隐层对状态的参数，整个模型共享 | $(N\times N)$  |
|    $\bm{V}$    |    输出层参数，整个模型共享    | $(M\times N)$  |
|    $\bm{b}$    |    隐层的偏置，整个模型共享    | $(N\times 1)$  |
|    $\bm{c}$    |    输出层偏置，整个模型共享    | $(M\times 1)$  |
|     $g()$      |         输出层激活函数         |       \        |
|     $f()$      |         隐层的激活函数         |       \        |

# 2 沿时间的反向传播算法
## 2.1 总体分析
首先快速总览一下RNN的全部流程。
1. 首先令模型的隐含状态$\bm{h^{(0)}=0}$。
2. 每一时刻$\bm{t}$的**输入$\bm{x^{(t)}}$** 都是一个向量(比如：在NLP中，可以使用词向量)，在经过模型后会得到**这一时刻的状态$\bm{h^{(t)}}$** 和**输出$\bm{y^{(t)}}$**。
3. 在NLP中，$\bm{y^{(t)}}$是由$\bm{s^{(t)}}$经过$g$ (通常为Softmax) 激活得到的，搭配Cross Entropy Loss (比如：在词表中挑选下一个单词，这是一个多分类问题) ，就能计算出此刻的损失$E^{(t)}$。
4. 计算出$E^{(t)}$后，并不能立即对模型参数进行更新。需要沿着时间$t$不断给出输入，计算出所有时刻的损失。模型总损失为$E=\sum_tE^{(t)}$
5. 我们需要根据总损失$E$计算所有参数的梯度$\frac{\partial E}{\partial \bm{U}},\frac{\partial E}{\partial \bm{W}},\frac{\partial E}{\partial \bm{V}},\frac{\partial E}{\partial \bm{b}},\frac{\partial E}{\partial \bm{c}}$，再使用基于梯度的优化方法进行参数更新。

这就是一轮完整的流程，本文要讨论的就是：如何计算RNN模型参数的梯度。

## 2.2 求$\frac{\partial E}{\partial \bm{V}}$

$$\frac{\partial E}{\partial \bm{V}}=\sum_t\frac{\partial E^{(t)}}{\partial \bm{V}}$$

由公式 $\bm{s^{(t)}}=\bm{Vh^{(t)}}+\bm{c}$ 和 $\bm{y^{(t)}}=g(\bm{s^{(t)}})$，很容易有:  
$$\begin{aligned}
    \frac{\partial E^{(t)}}{\partial V_{ij}}&=\frac{\partial E^{(t)}}{\partial s_i^{(t)}}\frac{\partial s_i^{(t)}}{\partial V_{ij}}\\
    &=\frac{\partial E^{(t)}}{\partial y_i^{(t)}}\frac{\partial y_i^{(t)}}{\partial s_i^{(t)}}\frac{\partial s_i^{(t)}}{\partial V_{ij}}\\
    &=\frac{\partial E^{(t)}}{\partial y_i^{(t)}}g'(s_i^{(t)})h_j^{(t)}
\end{aligned}$$

推广到矩阵形式，即:  
> $$\frac{\partial E}{\partial \bm{V}}=\sum_t[\frac{\partial E^{(t)}}{\partial \bm{y}}\odot g'(\bm{s^{(t)}})](\bm{h^{(t)}})^\mathrm{T}$$

## 2.3 求$\frac{\partial E}{\partial \bm{U}}$
