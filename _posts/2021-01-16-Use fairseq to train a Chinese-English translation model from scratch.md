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
提前组织一个目录结构的好处是可以让后面的一系列操作更加统一、规范化。下表中```~```代表linux系统中**我的用户目录**, v15news目录名代表此次我使用的数据集名称  
```php
~
├── mosesdecoder
├── subword-nmt
├── fairseq
└── nmt
    ├── data
        └── v15news
            └── data-bin        # 用于存放二进制文件
    ├── models                  # 用于保存过程中的model文件和checkpoint
        └── v15news
            └── checkpoints     # 保存checkpoints
    ├── utils                   # 一些其他工具
        ├── split.py            # 用于划分train,valid,test
        └── cut2.py             # 用于划分src,tgt
    └── scripts                 # 一些脚本
```

## 1.2 相关工具
除**jieba**是使用```pip install```安装外，其他几个工具都是建议直接克隆库到自己的用户目录中，方便使用其脚本(**moses**/**subword-nmt**)，或未来可能要自己拓展其中的模型(**fairseq**)
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

# 2 数据的准备
## 2.1 平行语料
对于有监督神经中英机器翻译，能够找到的语料如下：
1. [NEU nlp lab 开源语料](https://github.com/NiuTrans/NiuTrans.SMT/tree/master/sample-data) (10w，国内政治新闻领域)
2. [WMT新闻翻译任务News Commentary语料](http://www.statmt.org/wmt20/translation-task.html) (32w左右，国际新闻领域。其实News Commentary每年都有新闻数据集，但是基本没啥变化，每次在前一年的基础上加几百句，所以这里的链接直接指向最新的WMT20)
3. [NIST数据集](https://catalog.ldc.upenn.edu/LDC2010T21) (200w左右，需要购买)
4. [United Nations Parallel Corpus](https://conferences.unite.un.org/UNCORPUS/zh) (1500w左右，联合国文件领域)

我本人使用过语料1、3，其中3是跟已购买的师兄要的，不向外提供。其实初次训练建议使用语料1，规模小训练快，能够快速体验整个流程。当然，中英还有很多其他语料，见[参考资料1](https://chinesenlp.xyz/#/docs/machine_translation), [2](https://www.cluebenchmarks.com/dataSet_search.html)

## 2.2 数据预处理
### 2.2.1 数据格式
在本篇博客中，我准备使用WMT20新闻翻译任务的**news-commentary-v15语料**，放于以下位置：  
```python
...
└── nmt
    ├── data
        └── v15news     
            └── news-commentary-v15.en-zh.tsv
...
```
格式如下：  
```
1929 or 1989?	1929年还是1989年?
PARIS – As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.	巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。
At the start of the crisis, many people likened it to 1982 or 1973, which was reassuring, because both dates refer to classical cyclical downturns.	一开始，很多人把这次危机比作1982年或1973年所发生的情况，这样得类比是令人宽心的，因为这两段时期意味着典型的周期性衰退。
...
```

### 2.2.2 切分
首先，需要将以上文件分成标准格式，即源语言(raw.zh)、目标语言(raw.en)文件各一个，一行一句，附自己写的脚本(cut2.py)：
```python
import sys

'''
Usage: 
python cut2.py fpath new_data_dir
'''

def cut2(fpath, new_data_dir, nsrc='zh', ntgt='en'):
    fp = open(fpath, encoding='utf-8')
    src_fp = open(new_data_dir + 'raw.' + nsrc, 'w', encoding='utf-8')
    tgt_fp = open(new_data_dir + 'raw.' + ntgt, 'w', encoding='utf-8')
    for line in fp.readlines():
        tgt_line, src_line = line.replace('\n', '').split('\t')
        src_fp.write(src_line + '\n')
        tgt_fp.write(tgt_line + '\n')
    src_fp.close()
    tgt_fp.close()

if __name__ == '__main__':      
    cut2(fpath=sys.argv[1], new_data_dir=sys.argv[2], nsrc='zh', ntgt='en')
```
切分后在目录中如下格式存放：  
```python
...
└── nmt
    ├── data
        └── v15news     
            ├── news-commentary-v15.en-zh.tsv
            ├── raw.zh
            └── raw.en
...
```

### 2.2.3 normalize-punctuation
### 2.2.3 中文分词
### 2.2.4 tokenize
### 2.2.5 truecase
### 2.2.6 bpe
### 2.2.7 clean
### 2.2.8 split
另外，两个语言都需要按比例划分出训练集、测试集、开发集(所以共6个文件，为方便区分，直接以 'train.en', 'valid.zh' 这样的格式命名)，附自己写的脚本(split.py)：
```python
import random
import sys

'''
Usage:
python split.py src_fpath tgt_fpath new_data_dir
'''

def split(src_fpath, tgt_fpath, nsrc='zh', ntgt='en', ratio=(0.9, 0.05, 0.05), new_data_dir=''):
  src_fp = open(src_fpath, encoding='utf-8')
  tgt_fp = open(tgt_fpath, encoding='utf-8')
  
  src_train, src_test, src_val = open(new_data_dir + 'train.' + nsrc, 'w', encoding='utf-8'), \
    open(new_data_dir + 'test.' + nsrc, 'w', encoding='utf-8'), open(new_data_dir + 'valid.' + nsrc, 'w', encoding='utf-8')
  tgt_train, tgt_test, tgt_val = open(new_data_dir + 'train.' + ntgt, 'w', encoding='utf-8'), \
    open(new_data_dir + 'test.' + ntgt, 'w', encoding='utf-8'), open(new_data_dir + 'valid.' + ntgt, 'w', encoding='utf-8')
  
  src, tgt = src_fp.readlines(), tgt_fp.readlines()
  for s, t in zip(src, tgt):
      rand = random.random()
      if 0 < rand <= ratio[0]:
        src_train.write(s)
        tgt_train.write(t)
      elif ratio[0] < rand <= ratio[0] + ratio[1]:
        src_test.write(s)
        tgt_test.write(t)
      else:
        src_val.write(s)
        tgt_val.write(t)
  
  src_fp.close()
  tgt_fp.close()
  src_train.close()
  src_test.close()
  src_val.close()
  tgt_train.close()
  tgt_test.close()
  tgt_val.close()

if __name__ == '__main__':      
    split(src_fpath=sys.argv[1], tgt_fpath=sys.argv[2], nsrc='zh', ntgt='en', ratio=(0.95, 0.025, 0.025), new_data_dir=sys.argv[3])
```
最后，data/v15news目录中有如下数据：  
```python
...
└── nmt
    ├── data
        └── v15news     
            ...
            ├── test.en
            ├── test.zh
            ├── train.en
            ├── train.zh
            ├── valid.en
            └── valid.zh
...
```

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
4. Findings of the 2019 Conference on Machine Translation (WMT19)
5. The NiuTrans Machine Translation System for WMT18, WMT19, WMT20
6. Baidu Neural Machine Translation Systems for WMT19