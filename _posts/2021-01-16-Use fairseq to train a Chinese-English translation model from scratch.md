---
title: Use fairseq to train a Chinese-English translation model from scratch
date: 2021-01-16
tags:
- 机器翻译
---
# 前言
由于毕设是做神经机器翻译相关，所以先尝试一下神经机器翻译的整个流程是非常有必要的。故将训练有监督**中英NMT模型**的整个流程，包括工具和数据的准备、数据的预处理、训练及解码等过程，以及过程中遇到的问题和解决方案记录在此，以供后期回顾，也希望能够给予别人一些帮助。

# 1 目录结构及相关工具
## 1.1 目录结构
提前组织一个目录结构的好处是可以让后面的一系列操作更加统一、规范化。下表中```~```代表linux系统中**我的用户目录**  
```python
~
├── mosesdecoder
├── subword-nmt
├── fairseq
└── nmt
    ├── data        # 用于存放训练数据及二进制文件
    ├── models      # 用于保存模型的checkpoints
    ├── utils       # 一些其他工具
    └── scripts     # 一些脚本
```

## 1.2 相关工具
除**jieba**是使用```pip```安装外，其他几个工具都是建议直接克隆库到自己的用户目录中，方便使用其脚本(**moses**/**subword-nmt**)，或未来可能要自己拓展其中的模型(**fairseq**)
1. [Moses](https://github.com/moses-smt/mosesdecoder) (一个SMT工具，在这里只会用到一些预处理脚本，如：**tokenisation**, **truecasing**, **cleaning**), 这是[文档](http://www.statmt.org/moses/?n=Moses.Baseline)，安装指令如下：  
```
git clone https://github.com/moses-smt/mosesdecoder.git
```
2. [subword-nmt](https://github.com/rsennrich/subword-nmt) (使用BPE算法生成子词的预处理脚本)，安装指令如下：  
```
git clone https://github.com/rsennrich/subword-nmt.git
```
3. [jieba](https://github.com/fxsjy/jieba) (中文分词组件)，安装指令如下:  
```
pip install jieba
```
4. [fairseq](https://github.com/pytorch/fairseq) (一个基于PyTorch的序列建模工具), 这是[文档](https://fairseq.readthedocs.io/en/latest/index.html#)，安装指令如下：  
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

# 2 数据
## 2.2 平行语料
对于有监督神经中英机器翻译，能够找到的语料如下：
1. [NEU nlp lab 开源语料](https://github.com/NiuTrans/NiuTrans.SMT/tree/master/sample-data) (10w，国内政治新闻领域)
2. [WMT新闻翻译任务News Commentary语料](http://www.statmt.org/wmt20/translation-task.html) (32w左右，国际新闻领域。其实News Commentary每年都有新闻数据集，但是基本没啥变化，每次在前一年的基础上加几百句，所以这里的链接直接指向最新的WMT20)
3. [NIST数据集](https://catalog.ldc.upenn.edu/LDC2010T21) (200w左右，需要购买)
4. [United Nations Parallel Corpus](https://conferences.unite.un.org/UNCORPUS/zh) (1500w左右，联合国文件领域)

我本人使用过语料1、3，其中3是跟已购买的师兄要的，不向外提供。另外，其实初次训练建议使用语料1，训练快，能够快速体验整个流程。当然，中英还有很多其他语料，见[参考资料1](https://chinesenlp.xyz/#/docs/machine_translation),[2](https://www.cluebenchmarks.com/dataSet_search.html)

在本篇博客中，我准备使用WMT20新闻翻译任务的**news-commentary-v15语料**
## 2.3 数据预处理

### 2.3.1 

# 3 训练过程
## 3.1 训练
## 3.2 推理

# 4 问题集锦
## 4.1 fairseq框架相关
### 4.1.1 多GPU训练报错
在linux下使用fairseq训练命令，内容如下：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train ...
```
出现如下报错：
```
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
	Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
Traceback (most recent call last):
  ...
  site-packages/torch/multiprocessing/spawn.py", line 111, in join
    raise Exception(
Exception: process 2 terminated with exit code 1
```

**解决方案：**  
直接使用如下命令```export MKL_THREADING_LAYER=GNU```，再重新运行训练命令即可。具体原因见[this issue](https://github.com/pytorch/pytorch/issues/37377)

# 参考资料
1. [如何使用fairseq复现Transformer NMT](http://www.linzehui.me/2019/01/28/%E7%A2%8E%E7%89%87%E7%9F%A5%E8%AF%86/%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8fairseq%E5%A4%8D%E7%8E%B0Transformer%20NMT/)
2. [手把手教你用fairseq训练一个NMT机器翻译系统 - 胤风
](https://blog.csdn.net/moreaction_/article/details/107252080)
3. [FaceBook-NLP工具Fairseq漫游指南(1)—命令行工具 - ZhuNLP](https://zhuanlan.zhihu.com/p/194176917)