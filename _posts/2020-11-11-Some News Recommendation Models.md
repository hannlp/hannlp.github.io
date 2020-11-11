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

# 1. NPA: Neural News Recommendation with Personalized Attention
## 1.1 背景
## 1.2 模型
# 2. NAML: Neural News Recommendation with Attentive Multi-View Learning

# 3. LSTUR: Neural News Recommendation with Long- and Short-termUser Representations

# 4. NRMS: Neural News Recommendation with Multi-Head Self-Attention