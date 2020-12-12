---
title: Google Colab Usage Record
date: 2020-12-12
tags:
- 环境配置
---

# 前言
实验室服务器不让外网连了，发现没了算力啥都做不了。于是对比多个国内外算力平台，最终选择很良心的免费算力平台：Google Colab，记录一些使用技巧，以便今后科研之用。

# 1 初次使用Colab的配置
## 1.1 挂载Google云端硬盘
在cell中输入：
```python
from google.colab import drive
drive.mount('/content/drive')
```
运行后按提示操作即可

## 1.2 选择gpu
**菜单栏：** 代码执行程序 - 更改运行时类型 - 硬件加速器 - GPU  
选择GPU之后便可以用```!nvidia-smi```命令查看当前正使用的GPU。一般现在都是Tesla T4，很不错

## 1.3 选择相应库的版本
**例：** Colab默认的tensorflow版本是2.x的，如果需要使用1.x版本就需要手动切换，即在cell中输入以下代码：  
```
%tensorflow_version 1.x
```
运行后重启Colab即可。  
遇到没有的库，直接```!pip install```即可

## 1.4 切换到当前工作路径
**法1：** 使用```%cd```命令  
**法2：** 使用```os.chdir()```方法  
(右键一个文件或目录，即可看到**复制路径**的选项)

# 2 一些需要注意的地方
1. 在cell中，其他linux命令都可以通过在前面加```!```使用(如```!ls```)，但```cd```命令需要用```%cd```才可以
2. **目录名**中最好不要有空格。如果有空格，用到此目录名时需要在空格前加```\```进行转义

# 参考资料
1. [Colab配置: 使用gpu训练模型](https://blog.csdn.net/Augurlee/article/details/103019181)