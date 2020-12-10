---
title: Conda Usage Record
date: 2020-11-26
tags:
- 环境配置
---
# 前言
发现每次用conda的时候都要去百度找命令，故在此记录我最常用到的命令，以方便我或者其他人查找使用！
以下所有命令在**2020-11-26**被验证可用，请放心使用！
* **tip**:三击即可选中某一行命令

# 1 常用命令
## 1.1 Conda基本配置
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
```

附链接：[清华源](https://mirrors.tuna.tsinghua.edu.cn/) | [中科大源](https://mirrors.ustc.edu.cn/) | [上交源](https://mirrors.sjtug.sjtu.edu.cn/#/)

## 1.2 环境管理
```python
# 查看已存在的环境
conda env list

# 创建指定python版本的虚拟环境(如果只写python=2，那么conda会自动寻找2.x中的最新版本)
conda create -n test python=2.7

# 删除某个环境
conda remove -n test --all

# 完整复制某个环境
conda create -n test2 -clone test

# 进入虚拟环境(windows)
activate test

# 退出虚拟环境(windows)
conda deactivate
```
## 1.3 包管理
```python
# 在指定环境安装包并指定版本,如果不用-n指定环境名称，则被安装在当前活跃环境
conda install -n test package=x.x

# 查看当前环境的所有包
conda list
```

# 2 常出现的错误
## 2.1 anaconda-navigator 不能正常启动
如果错误提示中明显能够看到pyQt5相关条目，并且anaconda prompt可以运行，则说明核心模块安装正确，是UI(界面插件)的问题

**原因：** 界面插件损坏  
**解决方案：**  
>1. 直接删除%安装目录%\Lib\site-packages\pyQt5目录，以及所有包含‘pyQt5’的目录
>2. 进入cmd,输入pip install pyQt5

* 参考：[anaconda navigator 突然打不开有可能是什么原因？](https://www.zhihu.com/question/52136894)

## 2.2 tensorflow-gpu安装相关
### 2.2.1 tensorflow 1.14.0版本不匹配系列
直接```conda install tensorflow-gpu=1.14```，然后进入Python环境尝试导入，发现以下警告:
```
FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecate
```

**原因：** numpy版本不适应  
**解决方案：** numpy降级即可。参考：tf-1.14.0、np-1.17.1时出现报错，就将numpy改为```pip install numpy==1.16.0```即无Warning  
* 参考：[解决python调用TensorFlow时出现FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecate](https://blog.csdn.net/BigDream123/article/details/99467316)

另有以下报错：
```
'h5py.h5r.Reference' has no attribute '__reduce_cython__' 
```

**原因：** 预计是h5py版本不匹配  
**解决方案：** 降低h5py版本即可。```pip install h5py==2.8.0``` // in Windows 10  
* 参考：[Error using h5r>init h5py.h5r (line 145) AttributeError: type object 'h5py.h5r.Reference' has no attribute 'reduce_cython'](https://github.com/h5py/h5py/issues/1151)

# 参考资料
1. [conda简直神了[conda基本废了]](https://www.jianshu.com/p/47a536e6ee20)
2. [conda的安装与使用](https://www.jianshu.com/p/edaa744ea47d)(目前这篇文章也在持续更新)
3. [anaconda conda环境管理命令](https://blog.csdn.net/yimingsilence/article/details/79388205)