---
title: Machine Translation Notebook 
date: 2020-11-16
tags:
- 自然语言处理
- 机器翻译
---

# 前言
nlp实验室肖桐老师、朱靖波老师主编的[《机器翻译-统计建模与深度学习方法(第二版)》](https://opensource.niutrans.com/mtbook/index.html)前些天已经略读了一遍(自然是跳过了统计机器翻译那两章#手动狗头)，再回头看时发现有很多知识又已经忘了，现打算再次精读此书，将遇到的所有重要的、需要推导或记忆的知识记录在本篇blog里，防止遗忘！

# 1 一句话神经机器翻译
深度学习神经网络推动下产生的第三代机器翻译技术，无需设计规则、先验假设和特征工程，使用Encoder-Decoder框架对分布式表示的语言进行端对端映射，已成为时代主流。

>注：在SMT时代，会对翻译过程进行假设(称为隐藏结构假设)，比如：源语言和目标语言的词或短语序列间存在某种对齐关系

*另外：这句话是我读完这本书之后总结的，之后也会在重复的阅读过程中不断修改*

# 2 机器翻译的评价指标
分为**有参考答案评价**(人工打分、BLEU)和**无参考答案评价**
## 2.1 BLEU(*Bilingual Evaluation Understudy*)

采用$n$-$gram$匹配+短句惩罚的方式：

$$\begin{aligned}
    \mathrm{BLEU}&=\mathrm{BP}\cdot\mathrm{exp}(\sum_{i=1}^Nw_n\cdot\mathrm{logP}_n)\\
    \mathrm{BP}&=
\end{aligned}
$$

# 3 统计机器翻译部分

# 4 零碎的深度学习知识
## 4.1 稳定性训练
### 4.1.1 归一化
* 批量归一化(Batch Normalization)：沿着**批次**方向进行均值为0，方差为1的归一化。
* 层归一化(Layer Normalization)：归一化操作沿着**序列**方向进行，为避免序列上不同位置输出结果的不可比性。

### 4.1.2 残差网络(*Residual Networks*)
采用跨层连接的结构，有$\bm{x_{l+1}}=F(\bm{x_l})+\bm{x_l}$。在反向传播时，在$\bm{x_l}$处的梯度为：

$$\begin{aligned}
    \frac{\partial L}{\partial \bm{x_l}}&=\frac{\partial L}{\partial \bm{x_{l+1}}}\cdot\frac{\partial \bm{x_{l+1}}}{\partial \bm{x_l}}\\
    &=\frac{\partial L}{\partial \bm{x_{l+1}}}(1+\frac{\partial F(\bm{x_l})}{\partial \bm{x_l}})\\
    &=\frac{\partial L}{\partial \bm{x_{l+1}}}+\frac{\partial L}{\partial \bm{x_{l+1}}}\cdot\frac{\partial F(\bm{x_l})}{\partial \bm{x_l}}
\end{aligned}
$$

将后一层的梯度直接传递到前一层中，从而缓解了梯度经多层多次累乘造成的梯度消失问题。

# 5 神经语言模型
## 5.1 一种评价指标-困惑度(*Perplexity, PPL*)

# 6 机器翻译中的数据处理
## 6.1 子词切分
1. [深入理解NLP Subword算法：BPE、WordPiece、ULM](https://zhuanlan.zhihu.com/p/86965595)
2. [BPE系列之—— BPE算法](https://blog.csdn.net/qq_40240102/article/details/101843196)
