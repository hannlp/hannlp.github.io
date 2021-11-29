---
title: Mathematics Strengthening
date: 2020-11-01
tags: 基础知识
---

# 前言
通过更新这篇博客，巩固我的数学基础，同时提高我对LaTeX语法([数学公式的输入](https://www.lolimay.cn/2019/01/22/katex%E8%AF%AD%E6%B3%95%E6%B5%8B%E8%AF%95/))的熟练程度~

# 1 基础
## 1.1 充分必要条件
**充分必要条件**（英语：sufficient and necessary condition）简称为充要条件。

在逻辑学中：
* 当命题“若P则Q”为真时，P称为Q的**充分条件**，Q称为P的**必要条件**。

因此：
* 当命题“若P则Q”与“若Q则P”皆为真时，P是Q的**充分必要条件**，同时，Q也是P的**充分必要条件**。
* 当命题“若P则Q”为真，而“若Q则P”为假时，我们称P是Q的**充分不必要条件**，Q是P的**必要不充分条件**，反之亦然。

**充分**的含义是，一个命题A的成立足够保证另一个命题B的成立——如果我们知道A成立，那么我们可以“充分”认为B成立。**必要**的含义是，要使得某个命题B成立，我们必须要有A成立（因为A是B的推论，A的不成立将会否定B，所以把A称为B的必要条件）。

参考：[1](https://zh.wikipedia.org/wiki/%E5%85%85%E5%88%86%E5%BF%85%E8%A6%81%E6%9D%A1%E4%BB%B6), [2](https://www.zhihu.com/question/59367069/answer/164498925)


# 2 极限

|          等价无穷小           |                           泰勒展开                           |
| :---------------------------: | :----------------------------------------------------------: |
|         $sinx\sim x$          |                   $sinx=x-\frac{1}{3!}x^3$                   |
|  $1-cosx\sim \frac{1}{2}x^2$  |           $cosx=1-\frac{1}{2}x^2+\frac{1}{4!}x^4$            |
|         $e^x-1\sim x$         |           $e^x=1+x+\frac{1}{2}x^2+\frac{1}{3!}x^3$           |
|         $tanx\sim x$          |                   $tanx=x+\frac{1}{3}x^3$                    |
|   $\mathrm{arctan}x\sim x$    |             $\mathrm{arctan}x=x-\frac{1}{3}x^3$              |
|        $ln(1+x)\sim x$        | $ln(1+x)=x-\frac{1}{2}x^2+\frac{1}{3}x^3-\frac{1}{4}x^4+...$ |
| $(1+x)^\alpha-1\sim \alpha x$ | $(1+x)^\alpha=1+\alpha x+\frac{\alpha(\alpha-1)}{2!}x^2+...$ |

* **口诀**：指对连，三角断，三角对数隔一换。对数函数一二三，三角指数有感叹。[知乎:weiyinfu](https://www.zhihu.com/question/25627482/answer/332242473)

$1. 当x\to0时，f(x)=x-sin(ax)与g(x)=x^2ln(1-bx)等价无穷小，求a,b.$
解：

$$\begin{aligned}
    sin(ax)&=ax-\frac{1}{6}(ax)^3\\
    f(x)&=x-sin(ax)=(1-a)x+\frac{1}{6}(ax)^3\\
    ln(1-bx)&=-bx\\
    g(x)&=x^2ln(1-bx)=-bx^3\\
    \lim_{x\rightarrow 0}\frac{f(x)}{g(x)}&=1,故a=1,b=-\frac{1}{6}
\end{aligned}
$$

$2. 求lim_{x\to0}\frac{[sinx-sin(sinx)]sinx}{x^4}.$

>- $tip1:由泰勒展开:sinx=x-\frac{1}{3!}x^3+\frac{1}{5!}x^5-\frac{1}{7!}x^7+...$
>- $tip2:lim_{x\to0}(x-sinx)=\frac{1}{6}x^3(由tip1得)$
>- $tip3:整体代换:由tip2得,lim_{f\to0}(f-sinf)=\frac{1}{6}f^3(f为任意函数)$

解：

$$\begin{aligned}
    原式&=lim_{x\to0}\frac{sinx-sin(sinx)}{sin^3x} \\
    &=lim_{f\to0}\frac{f-sinf}{f^3}\\
    &=lim_{f\to0}\frac{\frac{1}{6}f^3}{f^3}\\
    &=\frac{1}{6}
\end{aligned}
$$

$3.当x\to0时，f(x)=3sinx-sin(3x)与cx^k等价无穷小，求c，k.$

解：

$$\begin{aligned}
    3sinx&=3(x-\frac{1}{6}x^3)=3x-\frac{1}{2}x^3\\
    sin(3x)&=3x-\frac{1}{6}(3x)^3=3x-\frac{9}{2}x^3\\
    上式-下式&=4x^3，故c=4，k=3
\end{aligned}
$$

> 佐天，有两个年轻人问我，韩老师发生肾么事了？我说怎么回事，塔门说，韩老师你不讲武德，为什么泰勒公式就展了两项就不展了？我说你不懂，你再展一下试试，我一说完他啪的一声就展开来了，很快啊！然后就是，一个$3\times\frac{1}{5!}x^5$，一个$\frac{1}{5!}(3x)^5$，我全防出去了，全防出去了啊。我说你这没用，你200多斤的$x^5$也折不动我$x^3$的一根手指头(高阶无穷小)。传统泰勒，自然是展到为止，谢谢朋友们！
