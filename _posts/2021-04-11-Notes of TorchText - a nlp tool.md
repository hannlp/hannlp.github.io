---
title: Notes of TorchText - a nlp tool
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

```python
from torchtext.legacy import data, datasets
SRC = data.Field(pad_token='<pad>', batch_first=True)
TGT = data.Field(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', batch_first=True)
test = datasets.TranslationDataset(path='./test', exts=('.en', '.de'), fields=(('src', SRC), ('trg', TGT)))
SRC.build_vocab(test)
TGT.build_vocab(test)
```
# 2 Field
在这里，我用大写字母如```Field```表示一个类，用小写字母如```field```表示一个对象，本文其它部分亦如此。
## 2.1 field.vocab
在使用```field.build_vocab()```后，相应的词表便建立好了，可以使用下面的几个函数或属性：
1. ```vocab.stoi[]```：使用vocab将单词转化为索引
2. ```vocab.itos[]```：使用vocab将索引转化为单词
3. ```vocab.freqs```：一个```collections.Counter```对象，统计了词表中单词的词频。可以使用```Counter```的所有方法

```python
word = 'the'
word_id = SRC.vocab.stoi[word]
word = SRC.vocab.itos[word_id]
print(word_id, word)
# Out: 2 the

SRC.vocab.freqs.most_common(5)
# Out: [('the', 3775), (',', 3050), ('.', 2796), ('of', 1697), ('to', 1682)]
```
## 2.2 field.pad()
将一批长度不同的句子用PAD填充到这批句子中最长的句子长度。
```python
src = [['I', 'love', 'you', 'china', '!']]
padded = SRC.pad(src); print(padded)
# Out: [['I', 'love', 'you', 'china', '!']]

srcs = [['I', 'love', 'you', 'china', '!'],
        ['China', ',', 'i', 'very', 'love', 'you', '!'],
        ['Chinese', ',', 'is', 'my', 'born', 'country', 'i', 'like', 'it']]
padded = SRC.pad(srcs); print(padded)
# Out: [['I', 'love', 'you', 'china', '!', '<pad>', '<pad>', '<pad>', '<pad>'], 
#       ['China', ',', 'i', 'very', 'love', 'you', '!', '<pad>', '<pad>'], 
#       ['Chinese', ',', 'is', 'my', 'born', 'country', 'i', 'like', 'it']]
```

## 2.3 field.numericalize()
使用field，将一批PAD后的句子数值化，即将单词转换成词典中对应的索引。
```python
src_tokens = SRC.numericalize(padded)
print(src_tokens)
# Out: tensor([[  46,  998,   77,    0, 1590,    1,    1,    1,    1],
#              [1381,    3,  584,  300,  998,   77, 1590,    1,    1],
#              [3497,    3,   12,  177,  883,  304,  584,  181,   27]])

print(list([SRC.vocab.itos[x] for x in src_tokens[i]] for i in range(len(src_tokens))))
# Out: [['I', 'love', 'you', '<unk>', '!', '<pad>', '<pad>', '<pad>', '<pad>'], 
#       ['China', ',', 'i', 'very', 'love', 'you', '!', '<pad>', '<pad>'], 
#       ['Chinese', ',', 'is', 'my', 'born', 'country', 'i', 'like', 'it']]
```


# 3 Dataset