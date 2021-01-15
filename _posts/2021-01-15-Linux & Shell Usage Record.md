---
title: Linux & Shell Usage Record
date: 2021-01-15
tags:
- 操作系统
---
# 前言
最近需要在linux系统上写一些脚本(bash文件)，方便批处理。自己目前的笔记本系统是windows10，现在使用的一套配置是Xshell6(使用服务器) + Xftp7(文件传输) + Notepad++(文件编辑)。用这篇博客记录使用过程中的一些问题及解决方案。

# 1 Linux命令记录
## 1.1 路径的表示 

# 2 问题记录
## 2.1 运行脚本时出现$'\r': 未找到命令
报错已经非常明确了，是linux无法解析$'\r'。这其实是windows与linux系统的差异导致的：因为linux上的换行符为\n，而windows上的换行符为\r\n，所以脚本到linux上就无法解析了([参考](https://blog.csdn.net/u010416101/article/details/80135293))。

**解决方案：**   
例如在windows下编辑好一个'hello.sh'文件，传输到了linux系统下，需要进行以下操作：
```
vi hello.sh
# 按'shift + :'进入命令模式
:set ff=unix
:wq
```