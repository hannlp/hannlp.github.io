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

# 
