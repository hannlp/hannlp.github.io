---
title: Python Strengthening
date: 2021-01-25
tags: 基础知识
---
# 前言
最近在读源码的过程中，发现有很多陌生的python知识，包括装饰器、上下文管理器等，还有很多非常有用但我没见过的自带模块。于是采用“步步高点读机”式学习方法，哪里不会搜哪里（狗头，并将进一步学习的过程记录在此。

# 0 推荐资源
## 0.1 学习资源
1. [Python documentation](https://docs.python.org/3/) - 官方文档，最好的学习资源
2. [Python Cookbook 3rd Edition Documentation](https://python3-cookbook.readthedocs.io/zh_CN/latest/index.html) - 是《Python Cookbook》的中文译本
3. [《Python进阶》](https://eastlakeside.gitbook.io/interpy-zh/) - 是《Intermediate Python》的中文译本

## 0.2 开发资源
1. [Google Python 风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents/)
2. [Abbreviations.com](https://www.abbreviations.com/https://www.abbreviations.com/) - 查单词缩写的网站，用于变量命名


# 1 Python进阶
## 1.1 装饰器

## 1.2 下划线的用法
### 1.2.1 在变量名中
![](https://pic3.zhimg.com/v2-cbc5c6037101c7d33cf0acd9f00a8cfa_r.jpg)

### 1.2.2 *_的用法
在元组拆包时，一般情况下，```=``` 左边的变量个数等于元组元素个数。

但如果只想使用元组中某几个元素的话，不需要的元素就没必要给它一个变量（因为这会占用内存），那就用 ```_``` 取代变量名。

如果不需要的元素是连续的，不用写多个 ```_``` ，直接写一个 ```*_``` 就行了。

另外，如果想把多个元素分配给一个变量p，可以使用 ```*p``` 。例子如下：  
```python
>>> a = (1, 2 ,3 ,4 ,5, 6)
>>> b, *_, d = a
>>> b, d
(1, 6)
>>> b, *c, d = a
>>> b, c, d
(1, [2, 3, 4, 5], 6)
```

# 2 常用模块
## 2.1 argparse
[官方文档](https://docs.python.org/zh-cn/3.7/library/argparse.html#module-argparse) | [简易教程](https://docs.python.org/zh-cn/3.7/howto/argparse.html)

## 2.2 typing
[Python中typing模块与类型注解的使用方法](https://www.jb51.net/article/166907.htm)

# 3 Python项目相关
## 3.1 requirements.txt
快速导出当前项目的类库生成requirements.txt：
```bash
pip install pipreqs
pipreqs .
```
安装requirements.txt中的类库
```bash
pip install -r requirements.txt
```

# 参考资料
1. [python中*_是什么意思？ - 薄荷红茶
](https://www.zhihu.com/question/374007342)
2. [Python中下划线的5种含义](https://zhuanlan.zhihu.com/p/36173202)
3. [python 项目自动生成requirements.txt文件](https://blog.csdn.net/Irving_zhang/article/details/79087569)