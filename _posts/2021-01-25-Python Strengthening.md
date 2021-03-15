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

|模式|举例|含义|
|:-:|:-:|:-:|
|单前导下划线|_var|命名约定，仅供内部使用。通常不会由Python解释器强制执行（通配符导入除外），只作为对程序员的提示|
|单末尾下划线|var_|按约定使用以避免与Python关键字的命名冲突|
|双前导下划线|__var|当在类内上下文中使用时，触发“名称修饰”。由Python解释器强制执行|
|双前导和双末尾下划线|\_\_var\_\_|表示Python语言定义的特殊方法。避免在自己的属性中使用这种命名方案|

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

## 2.3 Counter
Counter类在collections模块里，是字典的子类，可以为hashable对象计数。见([Python标准库——collections模块的Counter类](http://www.pythoner.com/205.html))
### 2.3.1 用法速览
1.创建(四种方法)
```python
c = Counter()  # 创建一个空的Counter类
c = Counter('gallahad')  # 从一个可iterable对象（list、tuple、dict、字符串等）创建
c = Counter({'a': 4, 'b': 2})  # 从一个字典对象创建
c = Counter(a=4, b=2)  # 从一组键值对创建
```

2.元素访问与删除
```python
c = Counter("aabc")
c["a"] # Out: 2
c["h"] # Out: 0
del c["a"]; c["a"] # Out: 0
```

3.常用操作
```python
sum(c.values())  # 所有计数的总数
c.clear()  # 重置Counter对象，注意不是删除
list(c)  # 将c中的键转为列表
set(c)  # 将c中的键转为set
dict(c)  # 将c中的键值对转为字典
c.items()  # 转为(elem, cnt)格式的列表
Counter(dict(list_of_pairs))  # 从(elem, cnt)格式的列表转换为Counter类对象
c.most_common()[:-n:-1]  # 取出计数最少的n-1个元素
c += Counter()  # 移除0和负值
```
### 2.3.2 常用函数
1.most_common(n)
返回一个TopN列表，如果n没有被指定，则返回所有元素。当多个元素计数值相同时，排列是无确定顺序的。
```python
# 使用TorchText统计词频的例子，其中freqs是Counter对象
SRC.vocab.freqs.most_common(6)
Out:
[('the', 3775),
 (',', 3050),
 ('.', 2796),
 ('of', 1697),
 ('to', 1682),
 ('a', 1303)]
```

# 3 Python项目相关
## 3.1 requirements.txt
快速导出当前项目的类库生成requirements.txt：
```bash
pip install pipreqs

# 在项目文件夹下
pipreqs ./
# 若报UnicodeDecodeError，则：
pipreqs ./ --encoding=utf-8
# 若想覆盖之前的requirements.txt，则：
pipreqs ./ --encoding=utf-8 --force
```

安装requirements.txt中的类库
```bash
pip install -r requirements.txt
```

# 参考资料
1. [python中*_是什么意思？ - 薄荷红茶
](https://www.zhihu.com/question/374007342)
2. [Python中下划线的5种含义](https://zhuanlan.zhihu.com/p/36173202)
3. [Python项目生成依赖包清单requirements .txt文件](https://zhuanlan.zhihu.com/p/57839415)