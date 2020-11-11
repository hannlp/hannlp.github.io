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

其中，DKN这篇论文我在去年已经读过并研究过代码了，现在效果比它好的有四个，NPA，NAML，LSTUR和NRMS。