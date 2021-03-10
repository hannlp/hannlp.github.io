---
title: Linux & Shell Usage Record
date: 2021-01-15
tags:
- 操作系统
---
# 前言
最近需要在linux系统上写一些脚本(bash文件)，方便批处理。自己目前的笔记本系统是windows10，现在使用的一套配置是Xshell6(使用服务器) + Xftp7(文件传输) + Notepad++(文件编辑)。用这篇博客记录使用过程中的一些问题及解决方案。

# 0 相关链接
1. [Linux / Shell 教程 - runoob](https://www.runoob.com/linux/linux-shell.html)
2. [Linux命令大全 - runoob](https://www.runoob.com/linux/linux-command-manual.html)
3. [一篇文章让你彻底掌握 Shell 语言 - 静默虚空](https://www.cnblogs.com/jingmoxukong/p/7867397.html)

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
当前在hans目录中，使用```ls /```列出根目录中的目录和文件：

```
bin   dev  home  lib64  mnt  proc  run   srv  tmp  var
boot  etc  lib   media  opt  root  sbin  sys  usr
```

当前在hans目录中，使用```ls .```列出hans目录中的目录和文件：

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

综上所述，想同时满足两个需求，可以直接使用```nohup command &```。更多的使用方法见[参考1](https://www.cnblogs.com/caodneg7/p/12028236.html), [2](https://mp.weixin.qq.com/s/nyT-FPdIUdJUiUCYVGEnTg)。下面给出一个完美的运行命令：  
```bash
nohup python -u train.py > train.log 2>&1 &
```

### 1.2.2 结束进程
结束进程最安全的方法是单纯使用kill命令，不加修饰符，不带标志，如```kill 32464 32465 32466 32467```(后面的几个数字是我要结束的进程号)，关于其他方式，见[参考](https://blog.csdn.net/lechengyuyuan/article/details/16337233)

### 1.2.3 查看进程
查看某个用户的所有进程：  
```bash
top -U 用户名
```

查看某一进程的详细信息：  
```bash
ps aux | grep 任务号
```

# 2 常用工具
## 2.1 Jupyter Lab
官方文档: [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/index.html) (需科学上网) , [IPython](https://ipython.readthedocs.io/en/stable/index.html)

### 2.1.1 快速部署
**目的:** 在实验室服务器后台运行jupyter lab服务，在自己电脑的浏览器上使用

1. 安装：```pip install jupyterlab```
2. 生成配置文件：```jupyter lab --generate-config```
3. 在```ipython```交互环境下：  
```bash
from jupyter_server.auth import passwd; passwd()
Enter password:         # 这个是进入网页jupyter lab的密码
Verify password: 
Out[1]: 'sha1:4907....' # 这个需要复制，一会用到
```
4. 修改配置文件(第2步生成的，默认在~/.jupyter/jupyter_lab_config.py，是一个隐藏文件)：  
```bash
c.ServerApp.ip = '*'
c.ServerApp.port = 8888                  # 端口号，不冲突即可
c.ServerApp.password = u'sha1:4907....'  # 刚才复制的
c.ServerApp.open_browser = False
```
5. 设置XShell隧道：```文件-(默认+当前)会话属性-隧道-TCP/IP转移规则``` 添加两个，一个拨出，一个传入。源主机填```localhost```，目标主机填写```服务器的ip地址```，端口号填写第4步配置文件中设置的的port(在这里即8888)。侦听端口任意设置，如8889  
6. 后台运行：```nohup jupyter lab > jupyter.log 2>&1 &```
7. 在自己电脑的浏览器上输入： ```localhost:8889```(端口即侦听端口)

### 2.1.2 使用技巧
1. **在一个ipynb中导入另一个ipynb的类：**  
```bash
%run OtherNotebook.ipynb
```
2. **关闭浏览器选项卡后，希望cell的输出不丢失：**  
```python
import sys
temp = sys.stdout
sys.stdout = open("my_log.txt", "a")
```
有很多思路([参考](https://www.thinbug.com/q/32539832))，但最高效的方法即将标准输出重定向到文件中。在cell运行时，可以查看文件中的输出，且关闭选项卡后cell仍会继续向文件追加输出。在运行完cell后，用以下代码恢复标准输出：  
```python
sys.stdout = temp
```


# 3 问题记录
## 3.1 运行脚本时出现```$'\r': 未找到命令```
报错已经非常明确了，是linux无法解析$'\r'。这其实是windows与linux系统的差异导致的：因为linux上的换行符为\n，而windows上的换行符为\r\n，所以脚本到linux上就无法解析了。见[参考](https://blog.csdn.net/u010416101/article/details/80135293)

**解决方案：**   
例如在windows下编辑好一个'hello.sh'文件，传输到了linux系统下，运行前需要进行以下操作：
```
vi hello.sh
# 按'shift' + ':'进入命令模式
:set ff=unix
:wq
```

## 3.2 生成nohup.out文件的内容始终是空的
使用```nohup python train.py &```命令时，生成的nohup.out文件始终是0kb

**原因：** python的输出有缓冲，导致out.log并不能够马上看到输出。  
**解决方案：** 加```-u```参数，使得python不启用缓冲。见[参考](https://blog.csdn.net/qq_31821675/article/details/78246808)

## 3.3 git clone报错
使用```git clone```命令时，出现如下报错:
```
Failed to connect to github.com/xx port 443: Timed out
```
**解决方案：** 输入如下命令
```bash
git config --global http.proxy http://127.0.0.1:1080
git config --global https.proxy http://127.0.0.1:1080
git config --global --unset http.proxy
git config --global --unset https.proxy
```

# 参考资料
1. [Linux - 路径的表示](https://blog.csdn.net/zhangzhebjut/article/details/22977477)
2. [服务器端jupyter notebook映射到本地浏览器的操作](https://www.qedev.com/python/271029.html)
3. [如何利用XShell隧道通过跳板机连接内网机器](https://jingyan.baidu.com/article/d5a880ebd69c2613f147ccbf.html)