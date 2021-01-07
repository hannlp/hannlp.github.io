---
title: Backward Propogation Through Time (BPTT) in RNN
date: 2020-12-14
tags:
- 深度学习
---

# 前言
RNN及其变种是一代经典，有必要认真学习。在推导了RNN的反向传播算法(BPTT)后，我发现一切反向传播算法都有普遍的规律：误差项都是有从后到前的递推关系的。另外，RNN按时间递推，其实与DNN中按层递推是非常相似的。遂将推导过程记录在此，方便回忆，也希望能给别人一点启发。

# 0 一点感悟
知乎上有这样一个问题：“Transformer会彻底取代RNN吗？”  
前几天简单看了下王者荣耀觉悟AI的论文，发现除了强化学习以外，也用到了熟悉的**LSTM**。我想，在觉悟AI这类**严格依赖于时间**(比如：每读一帧，就要立即进行相应的决策) 的情境中，好像就没法用Transformer这类基于self-attention的模型。因为self-attention的独特性使其**必须在一开始就知道所有时间位置的信息**，Transformer在NLP上的成功，我觉得还是因为**自然语言**并不算是严格依赖于时间的。因为我们在**数据中**看到的句子都是**完整的一句话**，这就方便了self-attention直接对每个位置进行建模。  
当然这只是我不成熟的思考(后面又想了想觉得有地方说的挺不对的)，还有其他重要的原因：比如Transformer、Bert这种基于self-attention结构的模型参数量极大，训练需要海量的数据。在数据量不足的情景，训练不充分，导致大bias，但相同的数据在RNN甚至LSTM上已经可以达到足够好的效果。

# 1 RNN模型结构及符号定义
## 1.1 模型结构
假设有一个时间序列$t=1,2,...,L$，在每一时刻$t$我们有：

$$\begin{aligned}
\bm{z^{(t)}}&=\bm{Ux^{(t)}}+\bm{Wh^{(t-1)}}+\bm{b}\\
    \bm{h^{(t)}}&=f(\bm{z^{(t)}})\\
    \bm{s^{(t)}}&=\bm{Vh^{(t)}}+\bm{c}\\
    \bm{y^{(t)}}&=g(\bm{s^{(t)}})
\end{aligned}$$

这就是RNN的结构。可以看到，每一时刻$t$的**隐含状态$\bm{h^{(t)}}$** 都是由**当前时刻的输入$\bm{x^{(t)}}$** 和**上一时刻的隐含状态$\bm{h^{(t-1)}}$** 共同得到的。下面是详细的符号定义：

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
3. 在NLP中，$\bm{y^{(t)}}$是由$\bm{s^{(t)}}$经过$g$ (通常为Softmax) 激活得到的，搭配Cross Entropy Loss (比如：在词表中挑选下一个单词，这是一个分类问题) ，就能计算出此刻的损失$E^{(t)}$。
4. 计算出$E^{(t)}$后，并不能立即对模型参数进行更新。需要沿着时间$t$不断给出输入，计算出所有时刻的损失。模型总损失为$E=\sum_tE^{(t)}$
5. 我们需要根据总损失$E$计算所有参数的梯度$\frac{\partial E}{\partial \bm{U}},\frac{\partial E}{\partial \bm{W}},\frac{\partial E}{\partial \bm{V}},\frac{\partial E}{\partial \bm{b}},\frac{\partial E}{\partial \bm{c}}$，再使用基于梯度的优化方法进行参数更新。

这就是一轮完整的流程，本文要讨论的就是：如何计算RNN模型参数的梯度。

## 2.2 求$\frac{\partial E}{\partial \bm{V}}$

$$\frac{\partial E}{\partial \bm{V}}=\sum_t\frac{\partial E^{(t)}}{\partial \bm{V}}$$

由公式 $\bm{s^{(t)}}=\bm{Vh^{(t)}}+\bm{c}$ 和 $\bm{y^{(t)}}=g(\bm{s^{(t)}})$，很容易有:

$$\begin{aligned}
    \frac{\partial E^{(t)}}{\partial V_{ij}}&=\frac{\partial E^{(t)}}{\partial s_i^{(t)}}\frac{\partial s_i^{(t)}}{\partial V_{ij}}\tag{a}\\
    &=\frac{\partial E^{(t)}}{\partial y_i^{(t)}}\frac{\partial y_i^{(t)}}{\partial s_i^{(t)}}\frac{\partial s_i^{(t)}}{\partial V_{ij}}\\
    &=\frac{\partial E^{(t)}}{\partial y_i^{(t)}}g'(s_i^{(t)})h_j^{(t)}
\end{aligned}$$

推广到矩阵形式，即:  
> $$\frac{\partial E}{\partial \bm{V}}=\sum_t[\frac{\partial E^{(t)}}{\partial \bm{y^{(t)}}}\odot g'(\bm{s^{(t)}})](\bm{h^{(t)}})^\mathrm{T}\tag{1}$$

## 2.3 求$\frac{\partial E}{\partial \bm{U}}$

$$\frac{\partial E}{\partial \bm{U}}=\sum_t(\frac{\partial E}{\partial \bm{U}})^{(t)}$$

> 细心的人会发现，与之前 ($\frac{\partial E}{\partial \bm{V}}=\sum_t\frac{\partial E^{(t)}}{\partial \bm{V}}$) 不同，这次的 **时间上标$^{(t)}$** 加在了括号外面。简单说一下原因：由于$\bm{V}$在输出层，所以它在每一时刻的梯度只与当前时刻的损失 $E^{(t)}$ 有关。但$\bm{U}$和$\bm{W}$在隐藏层，参与到了下一时刻的运算。在求它们每一时刻的梯度时，要使用总损失 $E$ 来表示。关于为什么**求和项**仍成立，可以参考[此文章](https://zybuluo.com/hanbingtao/note/541458)中的“数学公式超高能预警”部分，下同。

观察公式 $\bm{z^{(t)}}=\bm{Ux^{(t)}}+\bm{Wh^{(t-1)}}+\bm{b}$ 和 $\bm{h^{(t)}}=f(\bm{z^{(t)}})$，有：

$$\begin{aligned}
    (\frac{\partial E}{\partial U_{ij}})^{(t)}&=\frac{\partial E}{\partial z_i^{(t)}}\frac{\partial z_i^{(t)}}{\partial U_{ij}}\tag{b}\\
    &=\frac{\partial E}{\partial z_i^{(t)}}x_j^{(t)}
\end{aligned}$$

计算$\frac{\partial E}{\partial z_i^{(t)}}$这一项时，就需要仔细观察一下了。由于RNN的特性：计算$\bm{h^{(t)}}$时，同时需要 $\bm{x^{(t)}}$ 和 $\bm{h^{(t-1)}}$。所以 $\bm{z^{(t)}}$ 不仅会对当前时刻的输出造成影响，也会影响到下一时刻的输出，变量间具体的依赖关系如下图所示：

![](https://i.loli.net/2020/12/15/wMcLpmj7ZOqVSQh.png)

所以，$\frac{\partial E}{\partial z_i^{(t)}}$ 应该包含两部分：

$$\frac{\partial E}{\partial z_i^{(t)}}=\frac{\partial E^{(t)}}{\partial \bm{s^{(t)}}}\frac{\partial \bm{s^{(t)}}}{\partial z_i^{(t)}}+\frac{\partial E}{\partial \bm{z^{(t+1)}}}\frac{\partial \bm{z^{(t+1)}}}{\partial z_i^{(t)}}$$

前半部分：

$$\begin{aligned}
    &=\frac{\partial E^{(t)}}{\partial \bm{s^{(t)}}}\frac{\partial \bm{s^{(t)}}}{\partial z_i^{(t)}}\\
    &=\sum_k \frac{\partial E^{(t)}}{\partial s_k^{(t)}}\frac{\partial s_k^{(t)}}{\partial z_i^{(t)}}\\
    &=\sum_k \frac{\partial E^{(t)}}{\partial s_k^{(t)}}\frac{\partial s_k^{(t)}}{\partial h_i^{(t)}}\frac{\partial h_i^{(t)}}{\partial z_i^{(t)}}\\
    &=\sum_k \frac{\partial E^{(t)}}{\partial s_k^{(t)}}V_{ki}f'(z_i^{(t)})
\end{aligned}$$

后半部分：

$$\begin{aligned}
    &=\frac{\partial E}{\partial \bm{z^{(t+1)}}}\frac{\partial \bm{z^{(t+1)}}}{\partial z_i^{(t)}}\\
    &=\sum_k \frac{\partial E}{\partial z_k^{(t+1)}}\frac{\partial z_k^{(t+1)}}{\partial z_i^{(t)}}\\
    &=\sum_k \frac{\partial E}{\partial z_k^{(t+1)}}\frac{\partial z_k^{(t+1)}}{\partial h_i^{(t)}}\frac{\partial h_i^{(t)}}{\partial z_i^{(t)}}\\
    &=\sum_k \frac{\partial E}{\partial z_k^{(t+1)}}W_{ki}f'(z_i^{(t)})
\end{aligned}$$

带入原式，得到：

$$(\frac{\partial E}{\partial U_{ij}})^{(t)}=[\sum_k^M \frac{\partial E^{(t)}}{\partial s_k^{(t)}}V_{ki}+\sum_k^N \frac{\partial E}{\partial z_k^{(t+1)}}W_{ki}]\cdot f'(z_i^{(t)})\cdot x_j^{(t)}$$

> 引入**误差记号**，记 $\bm{\delta_y^{(t)}}=\frac{\partial E^{(t)}}{\partial \bm{s^{(t)}}},\bm{\delta_h^{(t)}}=\frac{\partial E}{\partial \bm{z^{(t)}}}$ 。再次提醒：某一时刻关于$\bm{s}$的误差只与当前时刻的损失有关，而关于$\bm{z}$的误差与后面的所有损失都有关。所以，还有以下关系：
> 
> $$\begin{aligned}
    \bm{\delta_y^{(t)}}&=\frac{\partial E}{\partial \bm{s^{(t)}}}=\frac{\partial E^{(t)}}{\partial \bm{s^{(t)}}}\\
    \bm{\delta_h^{(t)}}&=\frac{\partial E}{\partial \bm{z^{(t)}}}\not ={}\frac{\partial E^{(t)}}{\partial \bm{z^{(t)}}}
\end{aligned}$$

上式可改写为：

$$(\frac{\partial E}{\partial U_{ij}})^{(t)}=[\sum_k^M \delta_{y,k}^{(t)}V_{ki}+\sum_k^N \delta_{h,k}^{(t+1)}W_{ki}]\cdot f'(z_i^{(t)})\cdot x_j^{(t)}$$

推广到矩阵形式，即：

> $$\frac{\partial E}{\partial \bm{U}}=\sum_t [(\bm{V}^\mathrm{T}\bm{\delta_y^{(t)}}+\bm{W}^\mathrm{T}\bm{\delta_h^{(t+1)}})\odot f'(\bm{z^{(t)}})]\cdot \bm{x^{(t)}}\tag{2}$$

## 2.4 求$\frac{\partial E}{\partial \bm{W}}$

$$\frac{\partial E}{\partial \bm{W}}=\sum_t(\frac{\partial E}{\partial \bm{W}})^{(t)}$$

观察公式 $\bm{z^{(t)}}=\bm{Ux^{(t)}}+\bm{Wh^{(t-1)}}+\bm{b}$ ，有：

$$\begin{aligned}
    (\frac{\partial E}{\partial W_{ij}})^{(t)}&=\frac{\partial E}{\partial z_i^{(t)}}\frac{\partial z_i^{(t)}}{\partial W_{ij}}\tag{c}\\
    &=\frac{\partial E}{\partial z_i^{(t)}}h_j^{(t-1)}
\end{aligned}$$

可以发现公式$c$与公式$b$形式基本相同。所以很容易直接得出$\frac{\partial E}{\partial \bm{W}}$的矩阵形式：

> $$\frac{\partial E}{\partial \bm{W}}=\sum_t [(\bm{V}^\mathrm{T}\bm{\delta_y^{(t)}}+\bm{W}^\mathrm{T}\bm{\delta_h^{(t+1)}})\odot f'(\bm{z^{(t)}})]\cdot \bm{h^{(t-1)}}\tag{3}$$

## 2.4 引入$\bm{\delta_y^{(t)}}$与$\bm{\delta_h^{(t)}}$后发生了什么

之前我们一直在老老实实、循规蹈矩的计算参数的梯度。但回过头来重新审视一下公式$(a),(b),(c)$，会有一个惊人的发现： 

**其实我们并不需要推导到最后。**  

因为在$(a),(b),(c)$的第一行，早已经有了$\bm{\delta_y^{(t)}}$与$\bm{\delta_h^{(t)}}$的形式！我们只需要直接将其转化为矩阵表示就可以了。

所以，我们重写$\frac{\partial E}{\partial \bm{V}},\frac{\partial E}{\partial \bm{W}},\frac{\partial E}{\partial \bm{U}}$：

> $$\begin{aligned}
    \frac{\partial E}{\partial \bm{V}}&=\sum_t\frac{\partial E^{(t)}}{\partial \bm{s^{(t)}}}(\bm{h^{(t)}})^\mathrm{T}=\sum_t\bm{\delta_y^{(t)}}(\bm{h^{(t)}})^\mathrm{T}\\
    \frac{\partial E}{\partial \bm{U}}&=\sum_t\frac{\partial E}{\partial \bm{z^{(t)}}}(\bm{x^{(t)}})^\mathrm{T}=\sum_t\bm{\delta_h^{(t)}}(\bm{x^{(t)}})^\mathrm{T}\\
    \frac{\partial E}{\partial \bm{W}}&=\sum_t\frac{\partial E}{\partial \bm{z^{(t)}}}(\bm{h^{(t-1)}})^\mathrm{T}=\sum_t\bm{\delta_h^{(t)}}(\bm{h^{(t-1)}})^\mathrm{T}
\end{aligned}$$

所以说推导到最后，我们一切都白干了吗？

**当然不是。**

对比上式和$(1),(2),(3)$，我们可以找出$\bm{\delta_y^{(t)}}$与$\bm{\delta_h^{(t)}}$的计算方法：

$$\begin{aligned}
    \bm{\delta_y^{(t)}}&=\frac{\partial E^{(t)}}{\partial \bm{y^{(t)}}}\odot g'(\bm{s^{(t)}})\\
    \bm{\delta_h^{(t)}}&=(\bm{V}^\mathrm{T}\bm{\delta_y^{(t)}}+\bm{W}^\mathrm{T}\bm{\delta_h^{(t+1)}})\odot f'(\bm{z^{(t)}})
\end{aligned}$$

当然，如果使用$\mathrm{Softmax+CrossEntropy Loss}$这个组合，那么$\bm{\delta_y^{(t)}}$的形式会更为简洁。另外，我们可以看到$\bm{\delta_h^{(t)}}$这一项是可以**递推**计算的,这与DNN反向传播中的$\bm{\delta^l}$类似。所以，我们还需要计算最后一个时刻$L$的$\bm{\delta_h^{(L)}}$。因为他没有后一个递推项$\bm{\delta_h^{(L+1)}}$了，所以可以直接简化为：

$$\bm{\delta_h^{(L)}}=(\bm{V}^\mathrm{T}\bm{\delta_y^{(L)}})\odot f'(\bm{z^{(L)}})$$

在最后，再补上$\frac{\partial E}{\partial \bm{b}}$ 和 $\frac{\partial E}{\partial \bm{c}}$ 的推导：

> $$\begin{aligned}
    \frac{\partial E}{\partial \bm{b}}=\sum_t(\frac{\partial E}{\partial \bm{b}})^{(t)}=\sum_t\frac{\partial E}{\partial \bm{z^{(t)}}}\frac{\partial \bm{z^{(t)}}}{\partial \bm{b}}=\sum_t\frac{\partial E}{\partial \bm{z^{(t)}}}=\sum_t\bm{\delta_h^{(t)}}\\
    \frac{\partial E}{\partial \bm{c}}=\sum_t\frac{\partial E^{(t)}}{\partial \bm{c}}=\sum_t\frac{\partial E^{(t)}}{\partial \bm{s^{(t)}}}\frac{\partial \bm{s^{(t)}}}{\partial \bm{c}}=\sum_t\frac{\partial E^{(t)}}{\partial \bm{s^{(t)}}}=\sum_t\bm{\delta_y^{(t)}}
\end{aligned}$$

## 2.5 总结
总结下模型参数梯度的计算和更新流程，深刻感受下BPTT的魅力。  
1. 固定所有模型参数
2. 依次走过$L$个时刻，记录每一时刻的$\bm{x^{(t)}}$和$\bm{h^{(t)}}$，并得到每一时刻的损失$E^{(1)},E^{(2)},...E^{(L)}$，进而得到每一时刻的$\bm{\delta_y^{(1)}},\bm{\delta_y^{(2)}},...,\bm{\delta_y^{(L)}}$
3. 得到$\bm{\delta_y^{(L)}}$后，便可计算$\bm{\delta_h^{(L)}}$，进而递推向前计算每一时刻的$\bm{\delta_h^{(t)}}$
4. 得到所有$\bm{\delta_h^{(t)}}$后，便可计算所有模型参数的梯度
5. 更新所有模型参数

# 参考资料
1. [循环神经网络(RNN)模型与前向反向传播算法 - 刘建平Pinard](https://www.cnblogs.com/pinard/p/6509630.html)
2. [零基础入门深度学习(5) - 循环神经网络 - hanbingtao](https://zybuluo.com/hanbingtao/note/541458)
3. [学习笔记-循环神经网络(RNN)及沿时反向传播BPTT - 观海云远](https://zhuanlan.zhihu.com/p/61472450)