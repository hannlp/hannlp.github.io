---
title: Linux & Shell Usage Record
date: 2021-01-15
tags:
- 操作系统
---
# 前言
最近需要在linux系统上写一些脚本(bash文件)，方便批处理。自己目前的笔记本系统是windows10，现在使用的一套配置是Xshell6(使用服务器) + Xftp7(文件传输) + Notepad++(文件编辑)。用这篇博客记录使用过程中的一些问题及解决方案。

# 0 相关链接
1. [Shell 教程 - runoob](https://www.runoob.com/linux/linux-shell.html)
2. [Linux命令大全 - runoob](https://www.runoob.com/linux/linux-command-manual.html)

# 1 Linux使用记录
## 1.1 基础知识
### 1.1.1 Linux中路径的表示
可以使用```pwd```查看当前路径(从根目录开始)

**绝对路径：**  
Linux中，根目录从```/```开始

**相对路径：**  
```.``` 表示当前目录  
```..``` 表示上级目录  
```~``` 表示当前用户自己的家目录  
```~user``` 表示用户名为user的家目录，这里的user是在/etc/passwd中存在的用户名  

**举例：**  
* 当前在hans目录中，使用```ls /```列出根目录中的目录和文件：```[hanyuchen@IP-xxx hans]$ ls /``` ，得到:

```
bin   dev  home  lib64  mnt  proc  run   srv  tmp  var
boot  etc  lib   media  opt  root  sbin  sys  usr
```

* 当前在hans目录中，使用```ls .```列出hans目录中的目录和文件：```[hanyuchen@IP-xxx hans]$ ls .``` ，得到：

```
Anaconda3-2020.07-Linux-x86_64.sh  mycert.pem  mykey.key  test.py
```

## 1.2 常用命令
### 1.2.1 令进程在后台不挂断运行
**如下情景：**  
在自家笔记本上，使用Xshell登录实验室服务器运行某一进程，希望**1.该进程运行时我也能够使用其他指令**(如nvidia-smi)，且**2.关闭Xshell(ssh连接也会断)后进程依然能够运行**，这样我自己的笔记本就不用一直开机运行Xshell了

**解决方案：**  
对于需求1，可以使用```&```符号，例如```command &```。这个符号可以使进程**后台运行**，但是关闭终端(Xshell)后进程也会退出  
对于需求2，可以使用```nohup```命令，例如```nohup command```。这个命令是“no hang up”(不挂断)的缩写，可以使得关闭终端之后继续运行相应的进程

综上所述，想同时满足两个需求，可以直接使用```nohup command &```。更多的使用方法见[参考资料1](https://www.cnblogs.com/caodneg7/p/12028236.html), [2](https://mp.weixin.qq.com/s/nyT-FPdIUdJUiUCYVGEnTg)

### 1.2.2 关闭进程
关闭进程最安全的方法是单纯使用kill命令，不加修饰符，不带标志，如```kill 32464 32465 32466 32467```(后面的几个数字是我要关闭的进程号)，关于其他方式，见[参考资料](https://blog.csdn.net/lechengyuyuan/article/details/16337233)

# 2 问题记录
## 2.1 运行脚本时出现$'\r': 未找到命令
报错已经非常明确了，是linux无法解析$'\r'。这其实是windows与linux系统的差异导致的：因为linux上的换行符为\n，而windows上的换行符为\r\n，所以脚本到linux上就无法解析了([参考资料](https://blog.csdn.net/u010416101/article/details/80135293))。

**解决方案：**   
例如在windows下编辑好一个'hello.sh'文件，传输到了linux系统下，运行前需要进行以下操作：
```
vi hello.sh
# 按'shift' + ':'进入命令模式
:set ff=unix
:wq
```

# 参考资料
1. [Linux - 路径的表示](https://blog.csdn.net/zhangzhebjut/article/details/22977477)
2. 