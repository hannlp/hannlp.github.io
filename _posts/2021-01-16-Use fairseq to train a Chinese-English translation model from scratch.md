---
title: Use fairseq to train a Chinese-English translation model from scratch
date: 2021-01-16
tags:
- 机器翻译
---
# 前言
由于毕设是做神经机器翻译相关，所以先尝试一下神经机器翻译的整个流程是非常有必要的。故将在news-commentary-v15语料上训练有监督**中英NMT模型**的整个流程，包括工具和数据的准备、数据的预处理、训练及解码等过程，以及过程中遇到的问题和解决方案记录在此，以供后期回顾，也希望能够给予别人一些帮助。

# 1 相关工具及目录结构
## 1.1 相关工具
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

## 1.2 目录结构与初始化
### 1.2.1 目录结构
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

### 1.2.2 用于初始化的bash文件
这个文件是在上述目录结构的基础下，定义了一些后面需要用到的变量(主要是**各种脚本的路径**)，包括tokenizer.perl, truecase.perl等，可以在linux中使用bash xx.sh运行，也可以把这些内容直接全部复制到linux命令行中按回车  
```bash
#!/bin/sh

src=zh
tgt=en

SCRIPTS=~/mosesdecoder/scripts
TOKENIZER=${SCRIPTS}/tokenizer/tokenizer.perl
LC=${SCRIPTS}/tokenizer/lowercase.perl
TRAIN_TC=${SCRIPTS}/recaser/train-truecaser.perl
TC=${SCRIPTS}/recaser/truecase.perl
NORM_PUNC=${SCRIPTS}/tokenizer/normalize-punctuation.perl
CLEAN=${SCRIPTS}/training/clean-corpus-n.perl
BPEROOT=~/subword-nmt/subword_nmt

data_dir=~/nmt/data/v15news
model_dir=~/nmt/models/v15news
utils=~/nmt/utils
```
# 2 数据的准备
## 2.1 平行语料
对于有监督神经机器翻译，能够找到的中英平行语料如下：
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

### 2.2.3 切分
首先，需要将一个单独的数据文件切分成标准格式，即源语言(raw.zh)、目标语言(raw.en)文件各一个，一行一句，附自己写的脚本(~/nmt/utils/cut2.py)：
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
使用命令：  
```bash
python ${utils}/cut2.py ${data_dir}/news-commentary-v15.en-zh.tsv ${data_dir}/
```
切分后的文件在目录中如下格式存放：  
```python
├── data
    └── v15news     
        ├── news-commentary-v15.en-zh.tsv
        ├── raw.zh
        └── raw.en
```

### 2.2.4 normalize-punctuation(可选)
标点符号的标准化，同时对双语文件(raw.en, raw.zh)处理，使用命令：  
```bash
perl ${NORM_PUNC} -l en < ${data_dir}/raw.en > ${data_dir}/norm.en
perl ${NORM_PUNC} -l zh < ${data_dir}/raw.zh > ${data_dir}/norm.zh
```
处理后的文件在目录中如下格式存放：  
```python
├── data
    └── v15news     
        ...
        ├── norm.zh
        └── norm.en
```
效果如下:
```
# raw.en
“We can’t waste time,” he says.
Yet, according to the political economist Moeletsi Mbeki, at his core, “Zuma is a conservative.”

# norm.en
"We can't waste time," he says.
Yet, according to the political economist Moeletsi Mbeki, at his core, "Zuma is a conservative."
```

### 2.2.5 中文分词
对标点符号标准化后的中文文件(norm.zh)进行分词处理，使用命令：  
```bash
python -m jieba -d " " ${data_dir}/norm.zh > ${data_dir}/norm.seg.zh
```
处理后的文件在目录中如下格式存放：  
```python
├── data
    └── v15news     
        ...
        └── norm.seg.zh
```
效果如下:
```
# norm.zh
1929年还是1989年?
巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。
一开始，很多人把这次危机比作1982年或1973年所发生的情况，这样得类比是令人宽心的，因为这两段时期意味着典型的周期性衰退。

# norm.seg.zh
1929 年 还是 1989 年 ?
巴黎 - 随着 经济危机 不断 加深 和 蔓延 ， 整个 世界 一直 在 寻找 历史 上 的 类似 事件 希望 有助于 我们 了解 目前 正在 发生 的 情况 。
一 开始 ， 很多 人 把 这次 危机 比作 1982 年 或 1973 年 所 发生 的 情况 ， 这样 得 类比 是 令人 宽心 的 ， 因为 这 两段 时期 意味着 典型 的 周期性 衰退 。
```

### 2.2.6 tokenize
对上述处理后的双语文件(norm.en, norm.seg.zh)进行标记化处理(可以理解为将**英文单词**与**标点符号**用空格分开，同时将多个连续空格简化为一个空格)，使用命令：  
```bash
${TOKENIZER} -l en < ${data_dir}/norm.en > ${data_dir}/norm.tok.en
${TOKENIZER} -l zh < ${data_dir}/norm.seg.zh > ${data_dir}/norm.seg.tok.zh
```
处理后的文件在目录中如下格式存放：  
```python
├── data
    └── v15news     
        ...
        ├── norm.tok.en
        └── norm.seg.tok.zh
```
效果如下:
```
# norm.seg.zh
目前 的 趋势 是 ， 要么 是 过度 的 克制 （ 欧洲   ）   ，   要么 是 努力 的 扩展 （ 美国   ）   。
而 历史 是 不 公平 的 。   尽管 美国 要 为 当今 的 全球 危机 负 更 大 的 责任 ， 但 美国 可能 会 比 大多数 国家 以 更 良好 的 势态 走出 困境 。

# norm.seg.tok.zh
目前 的 趋势 是 ， 要么 是 过度 的 克制 （ 欧洲 ） ， 要么 是 努力 的 扩展 （ 美国 ） 。
而 历史 是 不 公平 的 。 尽管 美国 要 为 当今 的 全球 危机 负 更 大 的 责任 ， 但 美国 可能 会 比 大多数 国家 以 更 良好 的 势态 走出 困境 。

# norm.en
For geo-strategists, however, the year that naturally comes to mind, in both politics and economics, is 1989.
Of course, the fall of the house of Lehman Brothers has nothing to do with the fall of the Berlin Wall.

# norm.tok.en
For geo-strategists , however , the year that naturally comes to mind , in both politics and economics , is 1989 .
Of course , the fall of the house of Lehman Brothers has nothing to do with the fall of the Berlin Wall .
```

### 2.2.7 truecase
对上述处理后的英文文件(norm.tok.en)进行大小写转换处理(对于句中的每个英文单词，尤其是**句首单词**，在数据中**学习**最适合它们的大小写形式)，使用命令：  
```bash
${TRAIN_TC} --model ${model_dir}/truecase-model.en --corpus ${data_dir}/norm.tok.en
${TC} --model ${model_dir}/truecase-model.en < ${data_dir}/norm.tok.en > ${data_dir}/norm.tok.true.en
```
处理后的文件在目录中如下格式存放：  
```python
├── data
    └── v15news
        ...
        └── norm.tok.true.en
├── models
    └── v15news
        └── truecase-model.en
```
效果如下:
```
# norm.tok.en
PARIS - As the economic crisis deepens and widens , the world has been searching for historical analogies to help us understand what has been happening .
At the start of the crisis , many people likened it to 1982 or 1973 , which was reassuring , because both dates refer to classical cyclical downturns .
When the TTIP was first proposed , Europe seemed to recognize its value .
Europe is being cautious in the name of avoiding debt and defending the euro , whereas the US has moved on many fronts in order not to waste an ideal opportunity to implement badly needed structural reforms .

# norm.tok.true.en
Paris - As the economic crisis deepens and widens , the world has been searching for historical analogies to help us understand what has been happening .
at the start of the crisis , many people likened it to 1982 or 1973 , which was reassuring , because both dates refer to classical cyclical downturns .
when the TTIP was first proposed , Europe seemed to recognize its value .
Europe is being cautious in the name of avoiding debt and defending the euro , whereas the US has moved on many fronts in order not to waste an ideal opportunity to implement badly needed structural reforms .
```

### 2.2.8 bpe
对上述处理后的双语文件(norm.tok.true.en, norm.seg.tok.zh)进行子词处理(可以理解为更细粒度的分词)，使用命令：  
```bash
python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/norm.tok.true.en  -s 32000 -o ${model_dir}/bpecode.en --write-vocabulary ${model_dir}/voc.en
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.en --vocabulary ${model_dir}/voc.en < ${data_dir}/norm.tok.true.en > ${data_dir}/norm.tok.true.bpe.en

python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/norm.seg.tok.zh  -s 32000 -o ${model_dir}/bpecode.zh --write-vocabulary ${model_dir}/voc.zh
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.zh --vocabulary ${model_dir}/voc.zh < ${data_dir}/norm.seg.tok.zh > ${data_dir}/norm.seg.tok.bpe.zh
```
处理后的文件在目录中如下格式存放：  
```python
├── data
    └── v15news
        ...
        ├── norm.seg.tok.bpe.zh
        └── norm.tok.true.bpe.en
├── models
    └── v15news
        ...
        ├── voc.zh
        ├── voc.en
        ├── bpecode.zh
        └── bpecode.en
```
效果如下:
```
# norm.seg.tok.zh
从 一流 的 麻省理工学院 的 媒体 实验室 到 哈佛大学 的 数学 和 经济系 ， 亚洲 人 - 尤其 是 中国 和 印度人 - 到处 都 是 ， 犹如 公元前 一 世纪 在 雅典 的 罗马 人 一样 ： 他们 对 那里 学到 太 多 东西 的 人们 充满 了 敬佩 ， 而 他们 将 在 今后 几十年 打败 他们 学习 的 对象 。
这 不仅 加大 了 预防 危机 的 难度 - - 尤其 因为 它 为 参与者 提供 了 钻空子 和 逃避责任 的 机会 - - 还 使得 人们 越来越 难以 采取措施 来 应对 危机 。
它们 将 通胀 目标 设定 在 2 % 左右 - - 这 意味着 当 波涛汹涌 时 他们 根本 没有 多少 施展 空间 。

# norm.seg.tok.bpe.zh
从 一流 的 麻省理工学院 的 媒体 实验室 到 哈佛大学 的 数学 和 经济@@ 系 ， 亚洲 人 - 尤其 是 中国 和 印度人 - 到处 都 是 ， 犹如 公元前 一 世纪 在 雅典 的 罗马 人 一样 ： 他们 对 那里 学到 太 多 东西 的 人们 充满 了 敬佩 ， 而 他们 将 在 今后 几十年 打败 他们 学习 的 对象 。
这 不仅 加大 了 预防 危机 的 难度 - - 尤其 因为 它 为 参与者 提供 了 钻@@ 空子 和 逃避@@ 责任 的 机会 - - 还 使得 人们 越来越 难以 采取措施 来 应对 危机 。
它们 将 通胀 目标 设定 在 2 % 左右 - - 这 意味着 当 波@@ 涛@@ 汹涌 时 他们 根本 没有 多少 施展 空间 。

# norm.tok.true.en
indeed , on the surface it seems to be its perfect antithesis : the collapse of a wall symbolizing oppression and artificial divisions versus the collapse of a seemingly indestructible and reassuring institution of financial capitalism .
as a visiting professor at Harvard and MIT , I am getting a good preview of what the world could look like when the crisis finally passes .
one senses something like the making of an American-Asian dominated universe .

# norm.tok.true.bpe.en
indeed , on the surface it seems to be its perfect anti@@ thesis : the collapse of a wall symboli@@ zing oppression and artificial divisions versus the collapse of a seemingly inde@@ struc@@ tible and reassuring institution of financial capitalism .
as a visiting professor at Harvard and MIT , I am getting a good pre@@ view of what the world could look like when the crisis finally passes .
one senses something like the making of an American-@@ Asian dominated universe .
```

### 2.2.9 clean
对上述处理后的双语文件(norm.tok.true.bpe.en, norm.seg.tok.bpe.zh)进行过滤(取设定的**最小长度**和**最大长度**之间的句对，可有效过滤空白行)，使用命令：  
```bash
mv ${data_dir}/norm.seg.tok.bpe.zh ${data_dir}/toclean.zh
mv ${data_dir}/norm.tok.true.bpe.en ${data_dir}/toclean.en 
${CLEAN} ${data_dir}/toclean zh en ${data_dir}/clean 1 256
```
处理后的文件在目录中如下格式存放：  
```python
├── data
    └── v15news
        ...
        ├── clean.zh
        └── clean.en
```
效果如下(每行最开始标出了**行号**):
```python
# norm.tok.true.bpe.en
30 we can only hope that , in the end , the consequences of 2009 similarly prove to be far less dramatic than we now - intuitively and in our historical refle@@ xes - feel them to be .
31
32 one Hund@@ red Years of Ine@@ p@@ titude

# clean.en
30 we can only hope that , in the end , the consequences of 2009 similarly prove to be far less dramatic than we now - intuitively and in our historical refle@@ xes - feel them to be .
31 one Hund@@ red Years of Ine@@ p@@ titude
32 Berlin - The global financial and economic crisis that began in 2008 was the greatest economic stre@@ ss-@@ test since the Great Depression , and the greatest challenge to social and political systems since World War II .

# norm.seg.tok.bpe.zh
30 我们 只能 希望 2009 年 的 危机 同样 地 最后 被 证明 是 远远 低于 我们 现在 以 直觉 和 历史 回顾 的 方式 � � 感觉 到 的 那么 剧烈 。
31 
32 百年 愚@@ 顽

# clean.zh
30 我们 只能 希望 2009 年 的 危机 同样 地 最后 被 证明 是 远远 低于 我们 现在 以 直觉 和 历史 回顾 的 方式 � � 感觉 到 的 那么 剧烈 。
31 百年 愚@@ 顽
32 柏林 - - 2008 年 爆发 的 全球 金融 和 经济危机 是 自大 萧条 以来 最 严峻 的 一次 经济 压力 测试 ， 也 是 自 二战 以来 社会 和 政治 制度 所 面临 的 最 严重 挑战 。
```

### 2.2.10 split
最后，双语文件(clean.zh, clean.en)都需要按比例划分出训练集、测试集、开发集(所以共6个文件，为方便区分，直接以 'train.en', 'valid.zh' 这样的格式命名)，附自己写的脚本(split.py)：
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
使用命令：  
```bash
python ${utils}/split.py ${data_dir}/clean.zh ${data_dir}/clean.en ${data_dir}/
```
最后，data/v15news目录中有如下数据：  
```
├── data
    └── v15news
        ...
        ├── test.en
        ├── test.zh
        ├── train.en
        ├── train.zh
        ├── valid.en
        └── valid.zh
```

# 3 训练过程
## 3.1 生成词表及二进制文件
首先用预处理后的六个文件(train.zh, valid.en等)，调用```fairseq-preprocess```命令生成**词表**和**训练用的二进制文件**  
```
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${data_dir}/train --validpref ${data_dir}/valid --testpref ${data_dir}/test \
    --destdir ${data_dir}/data-bin
```
生成的文件都保存在data-bin目录中  
```
├── data
    └── v15news
        ...
        └── data-bin
            ├── dict.zh
            ├── dict.en
            ├── preprocess.log
            ├── train.zh-en.zh.idx
            ...
            └── valid.zh-en.en.bin
```

## 3.2 训练
使用```fairseq-train```命令进行训练，其中有很多可以自由设置的超参数，比如选择使用什么模型，模型的参数等。其中，```--save-dir``` 这个参数是指每一个epoch结束后模型保存的位置
```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup fairseq-train ${data_dir}/data-bin --arch transformer \
	--source-lang ${src} --target-lang ${tgt}  \
    --optimizer adam  --lr 0.001 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --max-tokens 4096  --dropout 0.3 \
    --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
    --max-update 200000  --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 --num-workers 8 \
	--save-dir ${model_dir}/checkpoints &
```
我自己训练时是在3块GTX TITAN X卡上跑了6个小时，共跑了49个epoch，但是在第22个epoch的时候已经收敛(只需要看验证集上的ppl的变化即可)
```
epoch 020 | valid on 'valid' subset | loss 4.366 | nll_loss 2.652 | ppl 6.29 | wps 50387.3 | wpb 8026 | bsz 299.8 | num_updates 14400 | best_loss 4.366
epoch 021 | valid on 'valid' subset | loss 4.36 | nll_loss 2.647 | ppl 6.27 | wps 51992.7 | wpb 8026 | bsz 299.8 | num_updates 15120 | best_loss 4.36
epoch 022 | valid on 'valid' subset | loss 4.361 | nll_loss 2.644 | ppl 6.25 | wps 49009.9 | wpb 8026 | bsz 299.8 | num_updates 15840 | best_loss 4.36
epoch 023 | valid on 'valid' subset | loss 4.369 | nll_loss 2.65 | ppl 6.28 | wps 51878.9 | wpb 8026 | bsz 299.8 | num_updates 16560 | best_loss 4.36
epoch 023 | valid on 'valid' subset | loss 4.369 | nll_loss 2.65 | ppl 6.28 | wps 51878.9 | wpb 8026 | bsz 299.8 | num_updates 16560 | best_loss 4.36
```
由于```--kep-last-epochs```这个参数我设为10，所以我最后10个epoch的模型都保存在以下目录中。此外，还会额外保存效果最好的模型(即第22个epoch)和最后一个模型(即第49个epoch，可以用于下一次训练)：  
```
├── models
    └── v15news
        ...
        └── checkpoints
            ├── checkpoint40.pt
            ...
            ├── checkpoint49.pt
            ├── checkpoint_best_.pt
            └── checkpoint_last.pt
```

## 3.3 解码
需要知道的是：训练阶段使用的是**训练集**和**验证集**，解码阶段使用的是**测试集**
```
fairseq-generate ${data_dir}/data-bin \
    --path ${model_dir}/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 8 > ./bestbeam8.txt
```

选取一部分结果，如下所示(**S**: 源句子，**T**: 目标句子，**H/D**: 预测的句子及其生成概率的log，句子质量越好，其生成概率越接近1，其log越接近0。**P**: 每一个词的生成概率的log。其中，$H=\frac{\sum P}{n}$)：
```
S-537	西班牙 的 人权 困境
T-537	Spain &apos;s Human-Rights Dilemma
H-537	-0.16863664984703064	Spain &apos;s Human Rights Quandary
D-537	-0.16863664984703064	Spain &apos;s Human Rights Quandary
P-537	-0.0973 -0.1385 -0.1464 -0.0123 -0.4252 -0.4299 -0.0110 -0.0884

S-5516	这是 不可 接受 的 。
T-5516	that is unacceptable .
H-5516	-0.35840675234794617	this is unacceptable .
D-5516	-0.35840675234794617	this is unacceptable .
P-5516	-0.7625 -0.5517 -0.2005 -0.1513 -0.1261

S-676	与 最初 版本 的 破产法 相 比较 ， 2006 年 的 法律 是 牢牢 扎根 于 市场经济 的 。
T-676	compared with the original bankruptcy code , the 2006 code is firmly rooted in the needs of a market economy .
H-676	-0.624997079372406	in contrast to the original bankruptcy law , the law of 2006 was firmly rooted in the market economy .
D-676	-0.624997079372406	in contrast to the original bankruptcy law , the law of 2006 was firmly rooted in the market economy .
P-676	-1.4995 -0.9434 -0.1292 -0.3479 -0.9758 -0.6600 -0.9037 -0.1836 -0.4983 -1.6406 -0.3142 -0.0344 -0.1685 -1.0289 -1.0286 -0.1917 -1.5369 -0.6586 -0.1119 -0.1333 -0.1361

S-432	用 缅因州 共和党 参议员 苏珊 · 柯林斯 （ Susan Collins ） 的话 说 ， 政府 关门 对 其 缅因州 阿卡迪亚 国家 公园 （ Acadia National Park ） 周边 &quot; 所有 小企业 都 造成 了 伤害 &quot; ， &quot; 这是 完全 错误 的 。 &quot; 是 她 首先 提出 了 和解 协议 纲要 并 送交 参议院 。
T-432	in the words of Senator Susan Collins , a Republican from Maine who first put together the outlines of a deal and took it to the Senate floor , the shutdown &quot; hurt all the small businesses &quot; around Acadia National Park in her home state , &quot; and that is plain wrong . &quot;
H-432	-0.7003933787345886	in the words of Susan Collins , a Republican senator from Maine , it would be a mistake to shut down the government &apos;s &quot; all small business &quot; around the Maine National Park , where she proposed a settlement and delivered it to the Senate .
D-432	-0.7003933787345886	in the words of Susan Collins , a Republican senator from Maine , it would be a mistake to shut down the government &apos;s &quot; all small business &quot; around the Maine National Park , where she proposed a settlement and delivered it to the Senate .
P-432	-1.2762 -0.3546 -0.0142 -0.1261 -0.0058 -0.7617 -0.1695 -0.2992 -0.0777 -0.3016 -0.4818 -0.0061 -0.0308 -0.3509 -2.5533 -1.5254 -0.2761 -1.1667 -0.6169 -0.6285 -1.2463 -0.0973 -1.4414 -0.3324 -0.2302 -0.3312 -0.6847 -1.0005 -0.1812 -2.9048 -0.3072 -1.8045 -0.0473 -0.8421 -0.4715 -0.6841 -1.1902 -1.6192 -0.3370 -2.3317 -0.3701 -0.2508 -3.0284 -0.2336 -1.1318 -0.3904 -0.1124 -0.0262 -0.2203 -0.1480
```
## 3.4 后处理及评价

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