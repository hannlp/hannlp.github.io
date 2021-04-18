---
title: Machine Translation Notebook 
date: 2020-11-16
tags:
- 自然语言处理
- 机器翻译
---

# 前言
nlp实验室肖桐老师、朱靖波老师主编的[《机器翻译-统计建模与深度学习方法(第二版)》](https://opensource.niutrans.com/mtbook/index.html)前些天已经略读了一遍(自然是跳过了统计机器翻译那两章#手动狗头)，再回头看时发现有很多知识又已经忘了，现打算再次精读此书，将遇到的所有重要的、需要推导或记忆的知识记录在本篇blog里，防止遗忘！

# 1 简述神经机器翻译
神经机器翻译是深度学习神经网络推动下产生的第三代机器翻译技术，相对于前两代基于规则、统计的机器翻译技术而言，它无需设计规则、先验假设，而是直接使用Encoder-Decoder框架对分布式表示（Distributed Representation）的语言进行端对端映射，具有模型结构统一、译文质量高、对问题建模更为直接等优势，已成为时代主流。

>注：在SMT时代，会对翻译过程进行假设(称为隐藏结构假设)，比如：源语言和目标语言的词或短语序列间存在某种对齐关系

# 2 机器翻译质量的评价
分为**有参考答案评价**(人工打分、BLEU)和**无参考答案评价**
## 2.1 BLEU(*Bilingual Evaluation Understudy*)

采用$n$-$gram$匹配+短句惩罚的方式

# 3 统计机器翻译部分

# 4 零碎的深度学习知识
## 4.1 稳定性训练
### 4.1.1 归一化
1. 批量归一化(Batch Normalization)：沿着**批次**方向进行均值为0，方差为1的归一化。
2. 层归一化(Layer Normalization)：归一化操作沿着**嵌入**方向进行

### 4.1.2 残差网络(*Residual Networks*)
采用跨层连接的结构，有$\bm{x_{l+1}}=F(\bm{x_l})+\bm{x_l}$。在反向传播时，在$\bm{x_l}$处的梯度为：

$$\begin{aligned}
    \frac{\partial L}{\partial \bm{x_l}}&=\frac{\partial L}{\partial \bm{x_{l+1}}}\cdot\frac{\partial \bm{x_{l+1}}}{\partial \bm{x_l}}=\frac{\partial L}{\partial \bm{x_{l+1}}}(1+\frac{\partial F(\bm{x_l})}{\partial \bm{x_l}})\\
    &=\frac{\partial L}{\partial \bm{x_{l+1}}}+\frac{\partial L}{\partial \bm{x_{l+1}}}\cdot\frac{\partial F(\bm{x_l})}{\partial \bm{x_l}}
\end{aligned}
$$

将后一层的梯度直接传递到前一层中，从而缓解了梯度经多层多次累乘造成的梯度消失问题。

# 5 神经语言模型
## 5.1 何为Neural Language Model
一个句子$\bm{s}=w_1w_2...w_m$，它存在的概率为

$$p(\bm{s})=p(w_m|w_1..w_{m-1})p(w_{m-1}|w_1...w_{m-2})...p(w_3|w_1w_2)p(w_2|w_1)$$

神经语言模型即使用神经网络模拟以上公式，循环神经网络能够很自然的做到这点

## 5.2 困惑度(Perplexity, PPL)
句子越好(概率大)，困惑度越小，也就是模型对句子越不困惑。困惑度计算方法为

$$PPL=p(w_1w_2...w_m)^\frac{1}{m}=\sqrt[m]{\frac{1}{p(w_1w_2...w_m)}}$$

语言模型中，句子的概率即所有词语生成概率的连乘，为了对长短句公平，所以开$m$次根号，也就是取几何平均数。(几何平均数的特点是，如果有其中的一个概率是很小的，那么最终的结果就不可能很大，从而要求好的句子的每个单词都要有基本让人满意的概率)

在不同模型中，其计算方法也不同。在NMT中，困惑度其实就是**交叉熵的指数形式**，也就是说，$PPL=e^{loss}$。推导见[参考1](https://zhuanlan.zhihu.com/p/114432097),[2](https://www.zhihu.com/question/58482430)

# 6 神经机器翻译模型
## 6.1 基于卷积神经网络的NMT模型
卷积神经网络（Convolutional Neural Network，CNN），由若干卷积层->非线性激活->池化层组成。首先在CV领域大量应用，也逐渐被拓展到NLP领域，用CNN实现NMT的最经典的一篇论文就是[Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)

### 6.1.1 CNN相对于全连接的特点
1. 全连接层考虑了所有的输入，层输出中的每一个元素都依赖于所有输入。但当处理图像这种以像素为单位的网格数据的时候，规模过大的数据会导致模型参数量过大。
2. 在一些网格数据中，通常具有**局部不变性**的特征，比如图像中不同位置的相同物体、语言序列中相同的n-gram等，全连接网络很难提取这些局部不变性特征。
3. CNN最大的特点在于具有**局部连接**（Locally Connected，也叫稀疏交互）和**权值共享**（Weight Sharing）的特性。卷积层中每个神经元只响应周围部分的局部输入特征，即稀疏交互。另外，卷积层使用相同的卷积核对不同位置进行特征提取，也就是采用权值共享来进一步减少参数量。

### 6.1.2 面向序列的卷积
背景：
1. NLP主要处理一维序列，如单词序列。序列长度往往不固定，而变长序列无法用固定大小的全连接网络进行直接建模。另外，过长的序列也会导致全连接网络参数量的急剧增加。
2. RNN劣势在于每一时刻的计算都依赖于上一时刻的结果，因此只能对序列进行串行处理，无法充分利用硬件设备进行并行计算。此外，在处理较长的序列时，这种串行的方式很难捕捉长距离的依赖关系。

优势：
1. CNN采用共享参数的方式处理固定大小窗口内的信息，且不同位置的卷积操作之间没有相互依赖，因此可以对序列进行高效地并行处理。
2. 针对序列中距离较长的依赖关系，可以通过堆叠多层卷积层来扩大感受野 (Receptive Field) ，这里感受野指能够影响神经元输出的原始输入数据区域的大小。

# 7 机器翻译中的数据处理
## 7.1 子词切分
1. [深入理解NLP Subword算法：BPE、WordPiece、ULM](https://zhuanlan.zhihu.com/p/86965595)
2. [BPE系列之—— BPE算法](https://blog.csdn.net/qq_40240102/article/details/101843196)

# 8 资源
## 8.1 平行语料
1. [联合国平行语料库](https://conferences.unite.un.org/UNCORPUS/zh)
2. [WMT19语料](http://www.statmt.org/wmt19/index.html)
3. [NiuTrans开源语料](https://github.com/NiuTrans/NiuTrans.SMT/tree/master/sample-data)
