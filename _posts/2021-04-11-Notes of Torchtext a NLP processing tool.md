---
title: Notes of TorchText a NLP processing tool
date: 2021-04-11
tags:
- 自然语言处理
---

# 前言

# 0 推荐资源
1. [TORCHTEXT DOCUMENTATION (0.8.1)](https://pytorch.org/text/0.8.1/) (官方文档，目前已更新到[0.9.0](https://pytorch.org/text/stable/index.html))
2. [pytorch/text](https://github.com/pytorch/text#data) (官方github仓库，其**readme**是一个非常简洁的使用指南)
3. [How to use TorchText for neural machine translation, plus hack to make it 5x faster](https://towardsdatascience.com/how-to-use-torchtext-for-neural-machine-translation-plus-hack-to-make-it-5x-faster-77f3884d95#8a90) (一个优质的使用torchtext预处理机器翻译数据的教程)

> **版本提示：** 在0.9.0版本中，之前版本的很多重要模块如```torchtext.data```、```torchtext.datasets```等已经移动到```torchtext.legacy```中了，导入时需要注意  

# 1 TorchText 0.8.1总体介绍


# 2 Field
## 2.1 field.vocab


## 2.2 field.pad()
使用field，将一批句子用PAD填充到这批句子中最长的句子长度。
```python
src = [['I', 'love', 'you', 'china', '!']]
padded = SRC.pad(src); print(padded)
# Out: [['I', 'love', 'you', 'china', '!']]

srcs = [['I', 'love', 'you', 'china', '!'],
        ['China', ',', 'i', 'very', 'love', 'you', '!'],
        ['Chinese', ',', 'is', 'my', 'born', 'country', 'i', 'like', 'it']]
padded = SRC.pad(srcs); print(padded)
# Out: [['I', 'love', 'you', 'china', '!', '<pad>', '<pad>', '<pad>', '<pad>'], ['China', ',', 'i', 'very', 'love', 'you', '!', '<pad>', '<pad>'], ['Chinese', ',', 'is', 'my', 'born', 'country', 'i', 'like', 'it']]
```

# 3 Dataset