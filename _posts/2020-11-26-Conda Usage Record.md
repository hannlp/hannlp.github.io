---
title: Conda Usage Record
date: 2020-11-26
tags:
- 环境配置
---
# 前言
发现每次用conda的时候都要去百度找命令，故在此记录我最常用到的命令，以方便我或者其他人查找使用！
以下所有命令在**2020-11-26**被验证可用，请放心使用！

# 1 常用命令
## 1.1 Conda基本配置
```python
# 更新conda

# 换国内源

```

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
# 为当前环境安装包

```

# 2 常出现的错误

# 参考资料
1. [conda简直神了[conda基本废了]](https://www.jianshu.com/p/47a536e6ee20)
2. 