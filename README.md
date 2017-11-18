# SVM demo 支持向量机小示例

# 示例说明
本示例是一个简单的试验，甚至连数据集都不用额外准备，旨在加深对SVM和核函数的理解，并看看如何利用 scikit-learn 中的svm，
编译环境是 jupyter notebook， 可以通过安装  Anaconda，导入 scikit-learn 库可以很容易实现，[github示例代码](https://github.com/youngxiao/SVM-demo)。本例中变没有用外部数据集，而是随机生成的点，大家在理解算法和 scikit-learn 熟练使用后，可以尝试导入有具体意义的数据集，看看SVM的效果。

## 概述
`SVM_demo_with_sklearn.ipynb`代码中主要分为两个部分
* 线性 SVM 分类器
* SVM 与核函数

## Section 1: 线性 SVM 分类器
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
支持向量机是解决分类和回归问题非常强大的有监督学习算法。简单说来，linear的SVM做的事情就是在不同类别的“数据团”之间划上一条线，对线性可分集，总能找到使样本正确划分的分界面，而且有无穷多个，哪个是最优？ 必须寻找一种最优的分界准则，SVM试图找到一条最健壮的线，什么叫做最健壮的线？其实就是离2类样本点最远的线。
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
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/reasult/svm1.png"/></div>

```
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X, y)

def plot_svc_decision_function(clf, ax=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            P[i, j] = clf.decision_function([xi, yj])
    # plot the margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

```
分隔超平面：上述将数据集分割开来的直线叫做分隔超平面。

超平面：如果数据集是N维的，那么就需要N-1维的某对象来对数据进行分割。该对象叫做超平面，也就是分类的决策边界。

间隔：一个点到分割面的距离，称为点相对于分割面的距离。数据集中所有的点到分割面的最小间隔的2倍，称为分类器或数据集的间隔。

最大间隔：SVM分类器是要找最大的数据集间隔。

支持向量：离分割超平面最近的那些点

sklearn的SVM里面会有一个属性support_vectors_，标示“支持向量”，也就是样本点里离超平面最近的点，组成的。
咱们来画个图，把超平面和支持向量都画出来。

```
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200, facecolors='none');
```
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/reasult/svm2.png"/></div>

可以用IPython的 `interact` 函数来看看样本点的分布，会怎么样影响超平面:
```
from IPython.html.widgets import interact

def plot_svm(N=100):
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=0, cluster_std=0.60)
    X = X[:N]
    y = y[:N]
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
    plt.xlim(-1, 4)
    plt.ylim(-1, 6)
    plot_svc_decision_function(clf, plt.gca())
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=200, facecolors='none')
    
interact(plot_svm, N=[10, 200], kernel='linear');
```
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/reasult/svm3.png"/></div>



## Section 2: SVM 与 核函数
对于非线性可切分的数据集，要做分割，就要借助于核函数了简单一点说呢，核函数可以看做对原始特征的一个映射函数，
不过SVM不会傻乎乎对原始样本点做映射，它有更巧妙的方式来保证这个过程的高效性。
下面有一个例子，你可以看到，线性的kernel(线性的SVM)对于这种非线性可切分的数据集，是无能为力的。
```
from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf);
```
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/reasult/svm4.png"/></div>

然后强大的高斯核/radial basis function就可以大显身手了:
```
r = np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2))

from mpl_toolkits import mplot3d

def plot_3D(elev=30, azim=30):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='spring')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

interact(plot_3D, elev=[-90, 90], azip=(-180, 180));
```
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/reasult/svm5.png"/></div>

你在上面的图上也可以看到，原本在2维空间无法切分的2类点，映射到3维空间以后，可以由一个平面轻松地切开了。
而带rbf核的SVM就能帮你做到这一点:
```
clf = SVC(kernel='rbf')
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200, facecolors='none');
```
<div align=center><img height="320" src="https://github.com/youngxiao/SVM-demo/raw/master/reasult/svm6.png"/></div>

## 关于SVM的总结:
* 非线性映射是SVM方法的理论基础，SVM利用内积核函数代替向高维空间的非线性映射；
* 对特征空间划分的最优超平面是SVM的目标，最大化分类边际的思想是SVM方法的核心；
* 支持向量是SVM的训练结果,在SVM分类决策中起决定作用的是支持向量。因此，模型需要存储空间小，算法鲁棒性强；
* 无任何前提假设，不涉及概率测度；
* SVM算法对大规模训练样本难以实施
* 用SVM解决多分类问题存在困难，经典的支持向量机算法只给出了二类分类的算法，而在数据挖掘的实际应用中，一般要解决多类的分类问题。可以通过多个二类支持向量机的组合来解决。主要有一对多组合模式、一对一组合模式和SVM决策树；再就是通过构造多个分类器的组合来解决。主要原理是克服SVM固有的缺点，结合其他算法的优势，解决多类问题的分类精度。如：与粗集理论结合，形成一种优势互补的多类问题的组合分类器。
* SVM是O(n^3)的时间复杂度。在sklearn里，LinearSVC是可扩展的(也就是对海量数据也可以支持得不错), 对特别大的数据集SVC就略微有点尴尬了。不过对于特别大的数据集，你倒是可以试试采样一些样本出来，然后用rbf核的SVC来做做分类。

## 依赖的 packages
* matplotlib
* pylab
* numpy
* seaborn

## 欢迎关注
* Github：https://github.com/youngxiao
* Email： yxiao2048@gmail.com  or yxiao2017@163.com
* Blog:   https://youngxiao.github.io
