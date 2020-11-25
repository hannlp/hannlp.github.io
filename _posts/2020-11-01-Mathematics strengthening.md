---
title: Mathematics strengthening
date: 2020-11-01
tags: 基础知识
---

# 前言
- 为什么写这个博客？
1. 大四上学期空余时间颇多，又因自身高数知识较薄弱，所以蹭了学院开的高数强化（学分课）。授课内容包括考研数学（高数，现代，概率论)重点题型及做题技巧。虽然课程以做题技巧为导向，但依旧可以收获颇多重要数学知识，例如多元函数的极限、连续、可微,以及使用泰勒展开求极限等。
2. 最近换了个更易于使用的博客平台，改变之前“各种折腾美化，却忽视更新博客内容”的策略。决定将更多的心思用在博客内容的创作上。其中，对于磕盐人员来说，数学公式的使用（Latex）的重要性不必多言，我也使自己的博客支持了数学公式(MathJax)。关于数学的博客必然少不了[数学公式的输入(Lolimay)](https://www.lolimay.cn/2019/01/22/katex%E8%AF%AD%E6%B3%95%E6%B5%8B%E8%AF%95/)，所以通过更新这篇博客，既可以提高我的数学基础，也可以提高我对LaTeX语法的熟练程度~
- 这个博客是什么样的形式？
大概只有两种形式，**题目&解答** 和 **部分重要定理总结**

---
# 1 极限一（计算）

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

> 佐天，有两个年轻人问我，韩老师发生肾么事了？我说怎么回事，塔门说，韩老师你不讲武德，为什么泰勒公式就展了两项就不展了？我说你不懂，你再展一下试试，我一说完他啪的一声就展开来了，很快啊！然后就是，一个$3\times\frac{1}{5!}x^5$，一个$\frac{1}{5!}(3x)^5$，我全防出去了，全防出去了啊。我说你这没用，你200多斤的$x^5$也折不动我$x^3$的一根手指头(高阶无穷小)。传统泰勒，自然是点到为止，谢谢朋友们！

$4.求\lim_{x\to0}\frac{\sqrt{1+2sinx}-x-1}{xln(1+x)}.$

$解：$

$5.求\lim_{x\to0}\frac{e^{x^2}-e^{2-2cosx}}{x^4}.$

$解：$

$6.当x\to0时，若x-tanx与x^k是同阶无穷小，则k=?.$

$解:$

# 2 极限二（证明）

# Updating ...