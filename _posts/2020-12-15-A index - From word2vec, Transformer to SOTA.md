---
title: A index - From word2vec, Transformer to SOTA 
date: 2020-12-15
tags:
- 自然语言处理
---

# 前言
之前看论文总想留下点什么痕迹(比如写个笔记)，但是发现有时候这也算是一种**造轮子**的行为，因为每个**划时代的研究**早就有无数人(包括**大牛**)写过总结，且已经总结的很好了。所以我反思了一下，这篇博客便诞生了，主旨是不为每篇经典单独造一个轮子，而是做一个**索引**，指向那些漂亮优秀的轮子(包括源码、总结等)。而我自己，只附上我对每一个模型的一段话总结(我希望每段话都是经过深思熟虑，总结到位的)。

# 1 词向量
## 1.1 word2vec
## 1.2 ELMo

# 2 预训练(语言)模型
## 2.1 Transformer
![](https://i.loli.net/2020/12/15/kUp6erNM2tAZ4zH.png)
![](https://i.loli.net/2020/12/15/ZGCuHEVtlbUd1ap.png)
### 2.1.1 一段话总结
### 2.1.2 模型源码
1. [attention-is-all-you-need-pytorch - jadore801120](https://github.com/jadore801120/attention-is-all-you-need-pytorch) (pytorch版本，**首推**，无其他冗余代码)
2. [The Annotated Transformer - harvardnlp](https://nlp.seas.harvard.edu/2018/04/03/attention.html) (pytorch版本，哈佛大学，有较多**注释**)
3. [tensor2tensor - tensorflow](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py) (tensorflow版本，**官方实现**，有较多冗余代码)

### 2.1.3 优质文章索引
1. [Attention? Attention! - Lilian Weng](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) (选读，主要介绍注意力机制)
2. [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/) (**首推**，以图像形式详细介绍了计算细节，例如block中的$QKV$矩阵运算、多头、前馈部分)
3. [Dissecting BERT Part 1: The Encoder - Miguel Romero Calvo](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3) (推荐，较上一篇更严谨，展示了**维度**的变化，并给出更多**真实的矩阵**示例)
4. [Dissecting BERT Appendix: The Decoder - Miguel Romero Calvo](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f) (**必读**，前两篇很少提及的**解码器**部分在这里详细介绍了，包括Mask、编码器与解码器间的attention、input_length与target_length转换)
5. [《Attention is All You Need》浅读（简介+代码）- 苏剑林](https://kexue.fm/archives/4765) (选读，对位置编码有所分析，也提出了transformer的一些不足)
6. [深入理解Transformer及其源码 - ZingpLiu](https://www.cnblogs.com/zingp/p/11696111.html) (选读，主要是在代码层面给予分析，与哈佛的文章类似)

## 2.2 Bert
![](https://i.loli.net/2020/12/15/18wZPMjQp5COuT2.png)
![](https://i.loli.net/2020/12/15/U4htoOYcn1kLTAy.png)
### 2.2.1 一段话总结
### 2.2.2 模型源码
1. [BERT-pytorch - codertimo](https://github.com/codertimo/BERT-pytorch) (pytorch版本，**首推**)
2. [bert - google-research](https://github.com/google-research/bert) (tensorflow版本，**官方实现**，附源码解析[1](https://www.cnblogs.com/Milburn/p/12031521.html),[2](https://blog.csdn.net/weixin_39470744))

### 2.1.3 优质文章索引
1. [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning) - Jay Alammar](https://jalammar.github.io/illustrated-bert/) (**首推**，站在NLP中Transfer Learning的角度，对比了ELMo，GPT等模型，重点讲解Bert，风格偏图像)
2. [Understanding BERT Part 2: BERT Specifics - Francisco Ingham](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73) (推荐，文中除Bert的训练细节外，还有很多自问自答、与GPT的对比、实验，便于更清楚的了解Bert)
3. [一文读懂BERT(原理篇) - 忧郁得茄子](https://blog.csdn.net/jiaowoshouzi/article/details/89073944) (推荐，一篇集大成的**中文**文章，引用了前几篇文章的很多内容，但总结的非常全面，涵盖了之前很少展开讲的**层归一化、padding mask**等，以及两种mask的结合方式)
4. [Bert时代的创新（应用篇）：Bert在NLP各领域的应用进展 - 张俊林](https://zhuanlan.zhihu.com/p/68446772) (选读，应用篇)
5. [关于BERT的若干问题整理记录 - Adherer](https://zhuanlan.zhihu.com/p/95594311) (选读，个人思考篇)