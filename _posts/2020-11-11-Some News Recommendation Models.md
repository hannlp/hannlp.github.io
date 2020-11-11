---
title: Some News Recommendation Models NPA/ NAML/ LSTUR/ NRMS
date: 2020-11-11
tags:
- 推荐系统
---

# 前言

上上上次组会研一学长汇报了一篇数据集文章：*MIND: A Large-scale Dataset for News Recommendation*，是微软为**新闻推荐**而发布的一个数据集。在听汇报时我发现这个数据集非常符合我的需求：

1. 首先，新闻推荐需要处理大量的**文本信息**，正与我未来方向(NLP)有较大关联
2. 新闻内容中包含着大量的实体，更有利于探索基于知识(知识图谱)的推荐方法

于是乎，我立马自己去读了这篇[MIND论文(点进去就是其官方网站)](https://msnews.github.io/)，数据格式等就暂不介绍，有兴趣的可以自己去官网查看。在论文中，微软官方实现了几个新闻推荐的算法，如下图:

![rec_models](/imgs/newsrec/rec_models.png)

其中，DKN这篇论文我在去年已经读过并研究过代码了，现在效果比它好的有四个，NPA，NAML，LSTUR和NRMS。我去找来并阅读了这四篇论文，发现第1，2，4篇是同一个人([清华一个很强的博士](https://wuch15.github.io/))发的...而且他也是第3篇的参与者。

本小菜鸡简单的在我的博客里写一下对这四篇论文的分析和理解~出场顺序大概就按照上图中的**效果从低到高**吧^ ^

# 0.几篇论文的共同点

因为这几篇论文的出处差不多，所以共同点非常多。

1. 这四篇论文都是基于三个主要模块：**新闻表示模型**、**用户表示模型**和**点击预测**(包括之前的DKN也是)。其中，*新闻表示模型*通常都从新闻内容(如新闻标题，新闻类别，新闻内容，新闻内容中包含的实体)中学习，而*用户表示*通常从用户的浏览历史新闻中学习，*点击预测*即使用新闻表示和用户表示来计算用户点击这个新闻的概率
2. 这四篇论文都大量使用了注意力机制(DKN在对用户建模时也使用了)

我在总结的时候，对于共同之处，我会一笔带过，重点关注每篇论文真正创新的地方~

# 1. NPA: Neural News Recommendation with Personalized Attention
## 1.1 核心思想

不同的用户会有不同的兴趣，同时每个用户往往有多种兴趣。所以，不同的用户可能会因为某个新闻的不同方面而点击这个新闻。

两个直观的感觉：

1. 一个新闻标题中的不同单词往往会对用户产生不同的影响
2. 并不是一个用户所浏览过的所有新闻都能反映他的偏好

基于这两个直觉，作者分别提出了*word-level Attention*的**news model**和*news-level Attention*的**user model**~

## 1.2 模型

![npa_model](/imgs/newsrec/npa_model.png)

### 1.2.1 新闻表示模型

新闻表示模型在上图中用绿色虚线圈着。

对于每一个**输入的新闻**(就是其标题文本，一个单词序列)，使用*新闻表示模型*得到最终的**表示向量**。过程大概如下：

1. 词嵌入。即使用词向量(word2vec/glove)技术，将标题中每个单词映射成其对应的向量表示，这样，新闻标题就变成了一个向量序列$E$
2. 卷积神经网络。使用卷积神经网络的初衷是想学习每个单词的**局部上下文信息**，也就是



# 2. NAML: Neural News Recommendation with Attentive Multi-View Learning

# 3. LSTUR: Neural News Recommendation with Long- and Short-termUser Representations

# 4. NRMS: Neural News Recommendation with Multi-Head Self-Attention