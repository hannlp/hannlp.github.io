---
title: Linux 使用记录
date: 2021-01-15
tags:
- 系统与环境
---
# 前言
目前我采用了轻薄本+服务器的科研模式，笔记本系统是windows10，使用的一套软件是Xshell6(连接服务器) + WinSCP(文件传输) + Notepad++(文件编辑)，将常用技巧记录如下。

# 0 相关链接
1. [Linux / Shell 教程](https://www.runoob.com/linux/linux-shell.html) & [Linux命令大全](https://www.runoob.com/linux/linux-command-manual.html) - runoob
2. [Linux命令大全(手册)](https://www.linuxcool.com/)
3. [一篇文章让你彻底掌握 Shell 语言 - 静默虚空](https://www.cnblogs.com/jingmoxukong/p/7867397.html)

# 1 基础知识
## 1.1 Linux中路径的表示
1. 可以使用```pwd```查看当前路径(从根目录开始)
2. **绝对路径：** Linux中，根目录从```/```开始
3. **相对路径：** ```.``` 表示当前目录，```..``` 表示上级目录，```~``` 表示当前用户自己的家目录，```~user``` 表示用户名为user的家目录，这里的user是在/etc/passwd中存在的用户名  

# 2 常用命令
## 2.1 进程相关
### 2.1.1 查看进程
1. 查看某个用户的所有进程：```top -U 用户名```
2. 查看某一进程的详细信息：```ps aux | grep 任务号```

### 2.1.2 结束进程
结束进程最安全的方法是单纯使用kill命令，不加修饰符，不带标志，如```kill 32464 32465 32466 32467```(后面的几个数字是我要结束的进程号)，关于其他方式，见[参考](https://blog.csdn.net/lechengyuyuan/article/details/16337233)

### 2.1.3 指定进程使用的gpu
1. 在终端执行程序时指定：```CUDA_VISIBLE_DEVICES=0,1```，当设为```-1```时代表不使用任何gpu
2. 在Python代码中指定：```import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"```

### 2.1.4 令进程在后台不挂断运行  
在自家笔记本上，使用Xshell登录实验室服务器运行某一进程，希望**1.该进程运行时我也能够使用其他指令**(如nvidia-smi)，且**2.关闭Xshell(ssh连接也会断)后进程依然能够运行**，这样我自己的笔记本就不用一直开机运行Xshell了

**解决方案：** 对于需求1，可以使用```&```符号，例如```command &```。这个符号可以使进程**后台运行**，但是关闭终端(Xshell)后进程也会退出。对于需求2，可以使用```nohup```命令，例如```nohup command```。这个命令是“no hang up”(不挂断)的缩写，可以使得关闭终端之后继续运行相应的进程

综上所述，想同时满足两个需求，可以直接使用```nohup command &```。更多的使用方法见[参考1](https://www.cnblogs.com/caodneg7/p/12028236.html), [2](https://mp.weixin.qq.com/s/nyT-FPdIUdJUiUCYVGEnTg)。下面给出一个完美的运行命令：  
```bash
nohup python -u train.py > train.log 2>&1 &
```

## 2.2 磁盘/文件相关
### 2.2.1 查看空间
查看整个服务器的空间
```bash
df -h
```
查看当前目录下每个一级目录的空间
```bash
du -h --max-depth=1
```

### 2.2.2 文件传输：scp
以下示例代表从10.10.10.10机器上的/opt/soft/中下载mongodb目录到本地的/opt/soft/目录来。```-r```代表递归的，用于传输目录
```
scp -r root@10.10.10.10:/opt/soft/mongodb /opt/soft/
```

### 2.2.3 解压
1.**zip文件：**```unzip```
注：使用unzip *.zip后，压缩包里有什么都原封不动的解压到当前文件夹。如果压缩包里不带文件夹，最好先新建一个文件夹如tmp，再unzip *.zip tmp/。这样控制着让文件夹不多也不少

2.**tar.gz文件：**```tar -zxvf```

### 2.2.4 内容抽取：cut
命令```cut```可以对file(或stdin)的每行抽取出希望抽取的部分([参考](https://man.linuxde.net/cut))。常用参数：  
1. ```-d```：指定分隔符，默认为“TAB”
2. ```-f```：抽取指定部分
3. ```N-```：从第N个部分到结尾
4. ```N-M```：从第N个部分到第M个（包括M）部分
5. ```-M```：从第1个部分到第M个（包括M）部分

**举例：** 在文件ldc_test.result中抽取出以```-T```开头的句子
```
-S: 决议 要求 埃塞俄比亚 立即 采取 具体 步骤 , 使 厄 埃 边界 委员会 能 在 没有 先决条件 的 情况 下 迅速 标@@ 定 边界 ; 要求 厄立特里亚 不再 拖延 , 不 设 先决条件 地 取消 对 埃@@ 厄 特派 团 的 行动 和 作业 的 所有 限制 .
-T: The resolution requires Ethiopia to immediately take concrete steps to allow the Erit@@ rea - Ethiopia Boundary Commission to speedily demarc@@ ate the border without any preconditions ; and requires Erit@@ rea to cancel all of its restrictions on UN@@ ME@@ E 's actions and operations without any further delay and without setting any preconditions .
-P: The resolution asked Ethiopia to take specific steps immediately to enable the Erit@@ rean border committee to rapidly set its boundary without a precondition ; Erit@@ rea would no longer delay or set a precedent for the removal of all restrictions on the actions and operations of the Erit@@ rean special missions .

-S: 有关 部门 应 强化 低 保@@ 户 在 享受 低 保 时 须 履行 的 义务 : 如 及时 通报 家庭 人员 及 收入 变化 情况 , 汇报 就业 情况 , 接受 定期 复@@ 审 等 , 而 有关 部门 则 应 加大 监督 检查 的 力度 .
-T: The relevant department should stress the obligations that welfare recipients must carry out while enjoying the welfare : for example , promptly notifying the changes in the family members and incomes , reporting the status of employment , accepting regular reviews , etc. On the other hand , the relevant department should step up monitoring and inspection .
-P: The relevant departments should strengthen the obligation of low - bonded households to carry out such tasks as helping low - income families to enjoy low - income insured : if timely reporting of changes in the income and changes in the income and reporting on employment , and receiving regular reviews , the relevant departments should intensify supervision and inspection .
```

使用命令：```grep ^-T ldc_test.result | cut -f2- -d" " > new.result```。其中，```-d" "```将空格设为分隔符，```-f2-```在用空格分开后的部分中，从第二个部分起，抽取后面所有部分。效果如下：

```
The resolution requires Ethiopia to immediately take concrete steps to allow the Erit@@ rea - Ethiopia Boundary Commission to speedily demarc@@ ate the border without any preconditions ; and requires Erit@@ rea to cancel all of its restrictions on UN@@ ME@@ E 's actions and operations without any further delay and without setting any preconditions .
The relevant department should stress the obligations that welfare recipients must carry out while enjoying the welfare : for example , promptly notifying the changes in the family members and incomes , reporting the status of employment , accepting regular reviews , etc. On the other hand , the relevant department should step up monitoring and inspection .
```

# 3 常用工具
## 3.1 Jupyter Lab
官方文档: [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/index.html) (需科学上网) , [IPython](https://ipython.readthedocs.io/en/stable/index.html)

### 3.1.1 快速部署
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

### 3.1.2 使用技巧
1.**内核(ipython kernel)管理：**
```bash
# 查看安装的kernel和其位置
jupyter kernelspec list
# 移除名为test的kernel
jupyter kernelspec remove test
```
仅使用conda新建一个环境，在jupyter lab中还无法使用其中的库。需要先配置相应的jupyter内核，并在菜单中选择此内核。见[参考](https://zhuanlan.zhihu.com/p/81605893)
```bash
# 建立环境并在环境中安装ipykernel
conda create -n 环境名称 python=3.7 ipykernel
# 将环境加入jupyter的kernel中
conda activate 环境名称
python -m ipykernel install --user --name 环境名称 --display-name "kernel在菜单中的名称"
```
据观察，在每个环境里都要分别安装jupyter lab，才能使用这个命令。但是配置应该只需要一次
```bash
python -m pip install jupyterlab
```

2.**在一个ipynb中导入另一个ipynb的类或变量：**  
```bash
%run OtherNotebook.ipynb
```
3.**关闭浏览器选项卡后，希望cell的输出不丢失：**  
```python
import sys
temp = sys.stdout
sys.stdout = open("my_log.txt", "a")
```
有很多思路([参考](https://www.thinbug.com/q/32539832))，但最高效的方法即将标准输出重定向到文件中。在cell运行时，可以查看文件中的输出，且关闭选项卡后cell仍会继续向文件追加输出。在运行完cell后，用以下代码恢复标准输出：  
```python
sys.stdout = temp
```
4.**linux命令中使用笔记本中的变量：**
其中```dir_path```是笔记本中直接定义的变量(python str)，有以下几种方式可以在linux命令中使用他们
```bash
dir_path = "/home/foo/bar"
!cp file1 $dir_path

dir_path = "/home/foo/bar"
!cp file1 {dir_path}

#sub_dir可以是字符串，总之和dir_path连接起来应该是一个完整的路径
!cp file1 {dir_path + sub_dir} 
```

## 3.2 git
### 3.2.1 问题记录
1.**使用```git clone```命令时，出现如下报错:**
```
Failed to connect to github.com/xx port 443: Timed out
```

**问题分析：** 代理没有设置好  
**解决方案：** 目前尚未完美解决，不过有两种方案：1.多次```clone```，会偶尔成功。2.将此仓库同步至gitee，再```clone``` gitee上的仓库地址即可。

2.**大小写不敏感**
例：在本地仓库建立一个Diagrams文件夹，push到了远程仓库。此时在本地把Diagrams修改为diagrams，再push，远程仓库依旧为Diagrams

**解决方案：**
在本地的项目文件夹输入```git config core.ignorecase false```，再push上去，发现远程仓库既有```Diagrams```又有```diagrams```。再使用```git rm -r --cached Diagrams``` 删除远程的Diagrams文件夹，再push上去，就好了。

# 4 问题记录
## 4.1 运行脚本时出现```$'\r': 未找到命令```
报错已经非常明确了，是linux无法解析$'\r'。这其实是windows与linux系统的差异导致的：因为linux上的换行符为\n，而windows上的换行符为\r\n，所以脚本到linux上就无法解析了。见[参考](https://blog.csdn.net/u010416101/article/details/80135293)

**解决方案：**   
例如在windows下编辑好一个'hello.sh'文件，传输到了linux系统下，运行前需要进行以下操作：
```
vi hello.sh
# 按'shift' + ':'进入命令模式
:set ff=unix
:wq
```

## 4.2 生成nohup.out文件的内容始终是空的
使用```nohup python train.py &```命令时，生成的nohup.out文件始终是0kb

**原因：** python的输出有缓冲，导致out.log并不能够马上看到输出。  
**解决方案：** 加```-u```参数，使得python不启用缓冲。见[参考](https://blog.csdn.net/qq_31821675/article/details/78246808)

# 参考资料
1. [Linux - 路径的表示](https://blog.csdn.net/zhangzhebjut/article/details/22977477)
2. [服务器端jupyter notebook映射到本地浏览器的操作](https://www.qedev.com/python/271029.html)
3. [如何利用XShell隧道通过跳板机连接内网机器](https://jingyan.baidu.com/article/d5a880ebd69c2613f147ccbf.html)