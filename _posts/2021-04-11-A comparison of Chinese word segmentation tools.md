---
title: A comparison of Chinese word segmentation tools
date: 2021-04-11
tags:
- 自然语言处理
---

# 前言
中文分词在汉语机器翻译系统中是一个关键部分，我在最近深有感触。本博客旨在对中文分词算法进行初步研究，并对现存的中文分词工具进行多方面的对比。（此博客不是那么紧急，打算慢慢更新）

# 1 问题发现
前阵子遇到了一个很奇怪的事情，自己的Transformer可以成功翻译“**我不爱你**”，“**我喜欢你**”，却总是把“**我爱你**”翻译错。一开始怀疑是训练数据不匹配、或者模型没有学到这句话的知识，当时并没有在意。
```python
Please input a sentence(zh): 我喜欢你。
I like you .
Please input a sentence(zh): 我不爱你。
I don &apos;t love you .
Please input a sentence(zh): 我爱你。
consider .
```

直到自己又实现并训练了基于LSTM的模型，发现依然无法翻译“**我爱你**”，才觉得这应该不是数据或模型的问题。偶然发现是分词这一步出错了。下面是我用jieba的分词情况：

```python
In [1]: import jieba
In [2]: print(list(jieba.cut('我喜欢你。')))
['我', '喜欢', '你', '。']

In [3]: print(list(jieba.cut('我不爱你。')))
['我', '不', '爱', '你', '。']

In [4]: print(list(jieba.cut('我爱你。')))
['我爱你', '。']
```

罪魁祸首jieba分词把“**我爱你**”这句话分成了一整个词。由于源语言词表中只有“**我**”，“**爱**”，“**喜欢**”，“**你**”这样的词，所以“**我爱你**”直接被当成一个UNK，所以模型才无法进行正确翻译。

# 2 分词算法
# 3 分词工具及对比