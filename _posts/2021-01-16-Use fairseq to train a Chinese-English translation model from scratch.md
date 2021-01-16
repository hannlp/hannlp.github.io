---
title: Use fairseq to train a Chinese-English translation model from scratch
date: 2021-01-16
tags:
- 机器翻译
---
# 前言

# 0 相关链接
1. [Moses](https://github.com/moses-smt/mosesdecoder)(preprocessing scripts to tokenisation,truecasing,cleaning), [here](http://www.statmt.org/moses/?n=Moses.Baseline) is documentation
2. [subword-nmt](https://github.com/rsennrich/subword-nmt)(preprocessing scripts to segment text into subword units)
3. [jieba](https://github.com/fxsjy/jieba)(中文分词组件)
4. [fairseq](https://github.com/pytorch/fairseq)(a sequence modeling toolkit), [here](https://fairseq.readthedocs.io/en/latest/index.html#) is documentation
# 1 数据
# 2 训练
# 3 推理

# 4 问题集锦
## 4.1 fairseq框架相关
### 4.1.1 多GPU训练报错
```
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
	Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
Traceback (most recent call last):
  ...
  site-packages/torch/multiprocessing/spawn.py", line 111, in join
    raise Exception(
Exception: process 2 terminated with exit code 1
```
