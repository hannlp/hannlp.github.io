---
title: Conda 使用记录
date: 2020-11-26
tags:
- 系统与环境
---
# 前言
这篇博客记录了一些conda使用方法，以及一些常用深度学习库安装、配置的踩坑过程

# 1 基本使用
以下所有命令在**2021-01-28**被验证可用！另附[官方文档](https://docs.conda.io/projects/conda/en/latest/index.html)

## 1.1 迅速安装
建议安装Miniconda，直接在[这里](https://docs.conda.io/en/latest/miniconda.html#linux-installers)找到对应的下载链接，右键复制链接，在服务器端用wget命令下载，bash命令安装。重启终端，即可使用conda命令。

## 1.2 基本配置
```python
# 更新conda至最新版本，也会更新其它相关包
conda update conda

# 添加清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/

# 设置搜索时显示通道地址
conda config --set show_channel_urls yes
# windows系统使用此命令后，会在你的C:\Users\用户名文件夹下会生成.condarc文件，在此文件里可以直接编辑(添加、删除)各种源

# 查看已经添加的源
conda config --get channels

# 恢复默认源
conda config --remove-key channels
```

附链接：[清华源](https://mirrors.tuna.tsinghua.edu.cn/), [中科大源](https://mirrors.ustc.edu.cn/), [上交源](https://mirrors.sjtug.sjtu.edu.cn/#/)

## 1.3 环境管理
在安装conda后，会发现命令提示符前多了一个(base)，这其实是conda自带的一个基础环境，而且每次登录后都会默认激活该环境：
```
(base) [hanyuchen@IP-xxx-xxx-xx-xx ~]$
``` 
当然，可以通过```conda deactivate```命令来退出(base)环境：
```
[hanyuchen@IP-xxx-xxx-xx-xx ~]$
```
如果希望每次登录默认为某一环境，可以在```~/.bash_profile```中添加```conda activate 该环境名```

可以发现，两个环境中的python版本不同，在不同环境中使用```pip/conda```安装的包也是相互隔离的。这方便我们运行依赖不同环境的代码，这就是conda的环境管理功能。常用命令如下：
```python
# 查看已存在的环境
conda env list

# 创建指定python版本的虚拟环境(如果只写python=2，那么conda会自动寻找2.x中的最新版本)
conda create -n test python=2.7

# 删除某个环境
conda remove -n test --all

# 完整复制某个环境
conda create -n test2 -clone test

# 进入虚拟环境(windows系统此命令可以省略前面的conda)
conda activate test

# 退出虚拟环境
conda deactivate
```

## 1.4 包管理
```python
# 在指定环境安装包并指定版本,如果不用-n指定环境名称，则被安装在当前活跃环境
conda install -n test package=x.x

# 查看当前环境的所有包
conda list
```

# 2 常出现的错误
## 2.1 python相关
### 2.1.1 启动时的interactivehook UnicodeDecodeError
具体报错如下：
```
Failed calling sys.__interactivehook__
Traceback (most recent call last):
  ...
  File "D:\My_Programs\Anaconda\lib\site-packages\pyreadline\lineeditor\history.py", line 82, in read_history_file
    for line in open(filename, 'r'):
UnicodeDecodeError: 'gbk' codec can't decode byte 0xaf in position 34: illegal multibyte sequence
```
**原因：** 记录历史命令的过程中遇到过中文，始终无法解码。具体分析过程见([参考](https://blog.csdn.net/hongxingabc/article/details/102610442))  
**解决方案：** 找到自己的‘history.py’文件，在读取文件的代码处添加'encoding=utf-8'，如下所示：

```python
for line in open(filename, 'r', encoding='utf-8'):
```
### 2.1.2 Win10系统Ipython异常，中止I/O操作
具体报错如下：
```
Unhandled exception in event loop:
  ...
Exception [WinError 995] The I/O operation has been aborted because of either a thread exit or an application request
Press ENTER to continue...
```
**解决方案：**```pip install --upgrade prompt-toolkit==2.0.10```即可。具体讨论见[this issue](https://github.com/ipython/ipython/issues/12049)

## 2.2 anaconda-navigator 不能正常启动
如果错误提示中明显能够看到pyQt5相关条目，并且anaconda prompt可以运行，则说明核心模块安装正确，是UI(界面插件)的问题

**原因：** 界面插件损坏  
**解决方案：**  
>1. 直接删除%安装目录%\Lib\site-packages\pyQt5目录，以及所有包含‘pyQt5’的目录
>2. 进入cmd,输入pip install pyQt5

* 参考：[anaconda navigator 突然打不开有可能是什么原因？](https://www.zhihu.com/question/52136894)

## 2.3 PyTorch安装相关
### 2.3.1 PyTorch与CUDA
安装PyTorch(gpu)最关键的就是要将**Pytorch版本**、**CUDA版本**以及**系统的驱动版本(driver version)** 三者匹配起来。([版本对应关系表](https://blog.csdn.net/weixin_42069606/article/details/105198845))

举个例子：
1. 我想使用1.6版本的PyTorch
2. 使用```nvcc -V```查看我现在的CUDA版本为10.1
3. 使用```nvidia-smi```查看到driver version为418.67
4. 在**版本对应关系表**中发现PyTorch1.6可以搭配CUDA10.1或9.2，但是CUDA10.1需要driver version>=418.96，我明显不满足(跑程序的时候也会提示设备版本too old)
5. 所以选择安装cuda9.2，在([官方提供的安装命令](https://pytorch.org/get-started/previous-versions/))中可以很容易的找到对应的安装命令如下：  
```
# CUDA 9.2
pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2.3.2 PyTorch检查CUDA是否可用
```python
import torch; print(torch.__version__); print(torch.cuda.is_available())
```

## 2.4 tensorflow-gpu安装相关
### 2.4.1 依赖的gpu环境
例：```tensorflow_gpu-1.14.0```需要安装```cuDNN:7.4，CUDA:10```。这是[经过测试的构建配置](https://tensorflow.google.cn/install/source_windows)

**查看CUDA版本：** 命令行输入```nvcc --version```  
**查看cuDNN版本：** 全局搜索'cudnn.h'，在最上方的几个宏定义处即显示其版本，例如下面所展示的就是7.6.5：  
```c
...
#if !defined(CUDNN_H_)
#define CUDNN_H_

#define CUDNN_MAJOR 7
#define CUDNN_MINOR 6
#define CUDNN_PATCHLEVEL 5
```
附：[Linux 和 Windows 查看 CUDA 和 cuDNN 版本](https://www.cnblogs.com/wuliytTaotao/p/11453265.html)

### 2.4.2 tf-1.14.0与其他库版本不匹配系列
直接```conda install tensorflow-gpu=1.14```，然后进入Python环境尝试导入，发现以下警告:
```
FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecate
```

**原因：** numpy版本不适应  
**解决方案：** numpy降级即可。例：tf-1.14.0、np-1.17.1时出现报错，就将numpy改为```pip install numpy==1.16.0```即可([参考](https://blog.csdn.net/BigDream123/article/details/99467316))

另有以下报错：
```
'h5py.h5r.Reference' has no attribute '__reduce_cython__' 
```

**原因：** 预计是h5py版本不匹配  
**解决方案：** 降低h5py版本即可。```pip install h5py==2.8.0``` ([参考](https://github.com/h5py/h5py/issues/1151))

## 2.5 网络相关
### 2.5.1 代理错误ProxyError
使用conda需要联网的命令时，出现以下报错：  
```
ProxyError: Conda cannot proceed due to an error in your proxy configuration.
Check for typos and other configuration errors in any '.netrc' file in your home directory,
any environment variables ending in '_PROXY', and any other system-wide proxy
configuration settings.
```

**原因：** 使用了科学上网软件，或者在公司代理下，但是没有对conda进行相关配置([参考](https://docs.anaconda.com/anaconda/user-guide/tasks/proxy/))  
**解决方案：**
windows下：在 "设置-网络和internet-代理-手动设置代理" 处查看**ip地址**和**端口号**，将其添加在.condarc文件中，形式如下：  
```
proxy_servers:
    http: http://username:password@corp.com:8080
    https: https://username:password@corp.com:8080
```

# 参考资料
1. [conda简直神了[conda基本废了]](https://www.jianshu.com/p/47a536e6ee20)
2. [conda的安装与使用](https://www.jianshu.com/p/edaa744ea47d)(目前这篇文章也在持续更新)
3. [anaconda conda环境管理命令](https://blog.csdn.net/yimingsilence/article/details/79388205)