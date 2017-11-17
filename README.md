# SVM demo 支持向量机小示例

# 示例说明
本示例是一个简单的试验，甚至连数据集都不用额外准备，旨在加深对SVM和核函数的理解，并看看如何利用 scikit-learn 中的svm，
编译环境是 jupyter notebook， 可以通过安装  Anaconda，导入 scikit-learn 库可以很容易实现，[github示例代码](https://github.com/youngxiao/SVM-demo)。本例中变没有用外部数据集，而是随机生成的点，大家在理解算法和 scikit-learn 熟练使用后，可以尝试导入有具体意义的数据集，看看SVM的效果。

## 概述
`SVM_demo_with_sklearn.ipynb`代码中主要分为两个部分
* 线性 SVM 分类器
* SVM 与核函数

## Section 1: 2D 线性 SVM 分类器
首先导入依赖包
```
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn; 
from sklearn.linear_model import LinearRegression
from scipy import stats
import pylab as pl
seaborn.set()
```
支持向量机是解决分类 和 回归 问题非常强大的有监督学习算法。简单说来，linear的SVM做的事情就是在不同类别的“数据团”之间划上一条线，用以分界，但是只划线是远远不够的，SVM试图找到一条最健壮的线，什么叫做最健壮的线？其实就是离2类样本点最远的线。
```
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')

# 其实随意给定3组参数，就可以画出3条不同的直线，但它们都可以把图上的2类样本点分隔开
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5);
```
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/result/SVM1.png"/></div>


## 依赖的 packages
* matplotlib
* pandas
* numpy
* seaborn

## 欢迎关注
* Github：https://github.com/youngxiao
* Email： yxiao2048@gmail.com  or yxiao2017@163.com
* Blog:   https://youngxiao.github.io
