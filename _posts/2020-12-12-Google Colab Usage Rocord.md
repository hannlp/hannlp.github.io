---
title: Google Colab Usage Record
date: 2020-12-12
tags:
- 系统与环境
---

# 前言
实验室服务器不让外网连了，发现没了算力啥都做不了。于是对比多个国内外算力平台，最终选择很良心的免费算力平台：Google Colab，记录一些使用技巧，以便今后科研之用。

# 1 初次使用Colab的配置
一般每天第一次打开Colab都需要重复如下操作：

## 1.1 选择gpu
**菜单栏 - 代码执行程序 - 更改运行时类型 - 硬件加速器 - GPU**  
选择GPU之后便可以用```!nvidia-smi```命令查看当前正使用的GPU。一般现在都是Tesla T4，很不错

## 1.2 挂载Google云端硬盘
既可以**直接鼠标点击挂载图标**，也可以按照如下方式挂载：  
在cell中输入：
```python
from google.colab import drive
drive.mount('/content/drive')
```
运行后按提示操作即可

## 1.3 选择相应库的版本
**例：** Colab默认的tensorflow版本是2.x的，如果需要使用1.x版本就需要手动切换，即在cell中输入以下代码：  
```
%tensorflow_version 1.x
```
运行后重启Colab即可。  
遇到没有的库，直接```!pip install```即可

## 1.4 切换到当前工作路径
右键一个文件或目录，即可看到**复制路径**的选项，有两种方法切换到该目录：  
**法1：** 使用```%cd```命令  
**法2：** 使用```os.chdir()```方法  

# 2 使用技巧
## 2.1 尤其要注意
1. 在cell中，其他linux命令都可以通过在前面加```!```使用(如```!ls```)，但```cd```命令需要用```%cd```才可以
2. **目录名**中最好不要有空格。如果有空格，用到此目录名时需要在空格前加```\```进行转义

## 2.2 其他使用技巧
1. 可以通过按 Ctrl 键，然后单击一个类名来跳转到**类定义**
2. 可以使用 ```!bash``` 命令使用交互式 shell
3. 可以同linux一样，使用 ```!nohup``` 命令，然后使用常规的 shell 命令，并在末尾添加 ```&``` 使其在后台运行。使用 ```!ps -ef``` 命令查看任务号，并用 ```!kill 任务号``` 的方式手动结束任务

# 3 深度学习环境相关
## 3.1 PyTorch 1.6.0 及环境配置
先运行完安装命令，并点击“RESTART RUNTIME”按钮
```
!pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```
检查CUDA是否可用
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

# 参考资料
1. [Colab配置: 使用gpu训练模型](https://blog.csdn.net/Augurlee/article/details/103019181)
2. [20种小技巧，玩转Google Colab](https://cloud.tencent.com/developer/article/1708477)
3. [Colab Pro 值得花 9.9$/mon 订阅吗？来看这篇完整评测- 佘城璐](https://zhuanlan.zhihu.com/p/145929375)
4. [Google Colab 的正确使用姿势 - 佘城璐](https://zhuanlan.zhihu.com/p/218133131)