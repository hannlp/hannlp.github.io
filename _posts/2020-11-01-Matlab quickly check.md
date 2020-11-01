---
title: Matlab基本操作
date: 2019-06-05 12:14:49
tags: 基本知识
---
## matlab与c的不同
1. 逻辑运算
或'|'，与'&'
2. 取整运算
取下整:floor(a)
取上整:ceil(a)
## 文件读取
1. 读取excel文件
```
xlsread('文件路径')
```
**matlab之导入EXCEL:错误，服务器出现意外情况**
打开EXCEL，在文件→选项→加载项里。
在下方管理中选中“com加载项”把复选框勾掉即可。
## 向量（一维数组）
* x的欧式长度
```
norm(x)
```
* 元素方差/标准差
```
var(x)/std(x)
```
* x与y的内积
```
dot(x,y)
```
* 距离
```
norm(x-y) or sqrt(sum((x-y).^2))
```
* 夹角
```
余弦值：dot(x,y)/(norm(x),norm(y))
弧度：acos(余弦值)
角度:弧度*(180/pi)
```
* 维数\最大值\最小值\平均值\元素和\积
```
length(x)\max(x)\min(x)\mean(x)\sum(x)\prod(x)
```
* 元素排序
```
从小到大：sort(x)
从大到小：sort(x,'descend')
```
* 元素位置
```
find(x[条件表达式])
```
* 向量的生成(以下都为行向量)
1. 以a为起点，k为步长,b为界
```
a:k:b
```
2. 等差数列:以a为起点，b为终点，均匀取N个元素
```
linspace(a,b,N)
```
3. 等比数列:以a为起点，b为终点，取N个元素

## 矩阵（二维数组）
* 矩阵的秩
```
rank(A)
```
* 矩阵的迹
```
trace(A) (A必须为方阵)
```
* 行列式
```
det(A) (A必须为方阵)
```
* 特征值/特征向量
```
[v,d]=eig(A) (A必须为方阵)
```
* 特征多项式
```
poly(A) (A必须是方阵)
```
* 逆
```
必须为方阵：inv(A)
不必须为方阵：pinv(A)
```
* **所有元素的**最大值\最小值\平均值
```
最大元素
max(max(A)) or max(A(:))
最小元素
min(min(A)) or min(A(:))
平均值
mean(A(:))
```
* **各列元素的**最大值\最小值\平均值
```
max(A)\min(A)\mean(A)
```
* 维数
```
size(A)
```
* （行/列）元素排序
```
各列元素排序(默认升序)
sort(A)
各行元素排序(默认升序)
sort(A,2)
各列元素排序(降序)
sort(A,2,'descend')
各行元素排序(降序)
sort(A,'descend')
```
* 将矩阵按列的顺序变为一列
```
A(:)
如:A(3)、A(5:8)、A([4,6,7,11])等操作均为在该列取相应元素
```
### diag（）用法
*核心：按照选择某一对角线上的所有元素*
* 矩阵的对角线转化为向量\向量转化为矩阵的对角线
```
diag(A)\diag(x)
```
* 矩阵副对角线的选择
```
例:diag(A,-2) or diag(A,1)
```
### 常用矩阵定义
* 单位矩阵
```
eye(n) or eye(m,n)
```
* 全0矩阵\全1矩阵
```
zeros(n) or zeros(m,n)\ones(n) or ones(m,n)
```
* 对角矩阵
```
diag(x)
```
* 取矩阵A的上（下）三角矩阵
```
triu(A)\tril(A)
```
* 魔方矩阵
```
magic()
百度百科:魔方矩阵又称幻方，是有相同的行数和列数，并在每行每列、对角线上的和都相等的矩阵。魔方矩阵中的每个元素不能相同。你能构造任何大小（除了2x2）的魔方矩阵。
```
* 随机矩阵（均匀分布/正态分布）
```
rand(n) or rand(m,n)\randn(n) or randn(m,n)
```
* 矩阵的翻转与变形
```
左右翻转:fliplr(A)
上下翻转:flipud(A)
逆时针旋转90°:rot90(A)
逆时针旋转180°:rot90(A,2)
顺时针旋转90°:rot90(A,-1)
```
* 矩阵的合并与变形
[合并](https://ww2.mathworks.cn/help/matlab/ref/repmat.html?s_tid=doc_ta)
[变形](https://ww2.mathworks.cn/help/matlab/ref/reshape.html?s_tid=doc_ta)
## 线性方程组(Ax=b or xA=b)
```
左除法:x=A\b or X=b/A
求逆法:x=inv(A)*b or pinv(A)*b
```
## 行列式
## 微积分
* 求极限
```
limit(expr,x,a)：当x=a时，对函数expr求极限，返回值为函数极限。
limit(expr)：默认当x=0时，对函数expr求极限，返回值为函数极限。
limit(expr,x,a,'left')：当x=a时，对函数expr求其左极限，返回值为函数极限。
limit(expr,x,a,'right')：当x=a时，对函数expr求其右极限，返回值为函数极限。
```
* [求导数](https://jingyan.baidu.com/article/e52e3615a01c8d40c70c517e.html)
* [级数求和](https://blog.csdn.net/u014147522/article/details/79078444)
* [泰勒展开](https://blog.csdn.net/wangh0802/article/details/73136329)

#绘图
