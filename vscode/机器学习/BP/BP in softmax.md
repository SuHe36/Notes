## BP in Softmax

### 1. softmax函数
设$X=[x_1,x_2,...,x_n]$, $Y=softmax(X)=[y_1,y_2,...,y_n]$,则：
$$y_i = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}$$
显然$\sum_{i=1}^{n}y_i = 1$

### 2.softmax函数求导$\frac{\partial y_i}{\partial x_j}$

1. 当$i==j$时：
$$\begin{aligned}
   \frac{\partial y_i}{\partial x_j} =& \frac{\partial}{\partial x_i}(\frac{e^{x_i}}{\sum_k e^{x_k}}) \\
    =& \frac{(e^{x_i})^{'}(\sum_k e^{x_k}) - e^{x_i}(\sum_k e^{x_k})^{'}}{(\sum_k e^{x_k})^{2}} \\
    =& \frac{e^{x_i}\cdot \sum_k e^{x_k} - e^{x_i}e^{x_i}}{(\sum_k e^{x_k})^{2}}  \\
    =& \frac{e^{x_i}\cdot \sum_k e^{x_k}}{(\sum_k e^{x_k})^{2}} - \frac{e^{x_i}e^{x_i}}{(\sum_k e^{x_k})^{2}} \\
    =& \frac{e^{x_i}}{\sum_k e^{x_k}} - \frac{e^{x_i}}{\sum_{k}e^{x_k}} \cdot \frac{e^{x_i}}{\sum_{k}e^{x_k}} \\
    =& y_i -y_i\cdot y_i \\
    =& y_i(1-y_i)
\end{aligned}$$

2. 当$i \neq j$时：
$$\begin{aligned}
    \frac{\partial y_i}{\partial x_j} =& \frac{\partial}{\partial x_i}(\frac{e^{x_i}}{\sum_k e^{x_k}}) \\
    =& \frac{(e^{x_i})^{'}(\sum_k e^{x_k}) - e^{x_i}(\sum_k e^{x_k})^{'}}{(\sum_k e^{x_k})^{2}} \\
    =& \frac{0\cdot \sum_k e^{x_k} - e^{x_i}e^{x_j}}{(\sum_k e^{x_k})^{2}}  \\
    =& \frac{- e^{x_i}e^{x_j}}{(\sum_k e^{x_k})^{2}} \\
    =& -\frac{e^{x_i}}{\sum_k e^{x_k}} \cdot \frac{e^{x_j}}{\sum_k e^{x_k}} \\
    =& -y_i\cdot y_j
\end{aligned}$$

综上所述：
$$\frac{\partial y_i}{\partial x_j} = 
\begin{cases}
    y_i - y_iy_i, \quad &当i=j; \\
    0 - y_i \cdot y_j, &当i \neq j;
\end{cases}$$

所以$\frac{\partial Y}{\partial X} = diag(Y) - Y^T\cdot Y$（当Y的shape为(1,n)时）。

### 3.softmax函数与交叉熵代价函数
$$C = -\sum_{k} \hat{y_k}logy_k$$
这里的$\hat{y_k}$是真实值，是训练的目标，为0或者1，在求导的时候是常量。
$y_k$是softmax函数的输出值，是训练结果。
$$\begin{aligned}
    \frac{\partial C}{\partial x_k} =& \sum_{i}\frac{\partial C}{\partial y_i} \cdot \frac{\partial y_i}{\partial x_k} \\
    =& -\sum_{i} \frac{\partial}{\partial y_i}(\hat{y_i}logy_i)\cdot \frac{\partial y_i}{\partial x_k} \\
    =& -\sum_{i}\frac{\hat{y_i}}{y_i} \cdot \frac{\partial y_i}{\partial x_k} \\
    =& -\frac{\hat{y_k}}{y_k}\cdot \frac{\partial y_k}{\partial x_k} - \sum_{i \neq k}\frac{\hat{y_i}}{y_i}\cdot \frac{\partial y_i}{\partial x_k} \\
    =& -\frac{\hat{y_k}}{y_k}\cdot y_k(1-y_k) - \sum_{i \neq k}\frac{\hat{y_i}}{y_i}(-y_iy_k) \\
    =& -\hat{y_k} + \hat{y_k}y_k + \sum_{i \neq k} \hat{y_i}y_k \\
    =& -\hat{y_k} + y_k\sum_{i}\hat{y_i} \\
    =& -\hat{y_k} + y_k \\
    =& y_k - \hat{y_k}
\end{aligned}$$

log似然代价函数$C$对每一个$x_i$求偏导，结果都是：
$$\frac{\partial C}{\partial x_i} = y_i - \hat{y_i}$$
即：
$$\frac{\partial C}{\partial X} = Y - \hat{Y}$$

### 4.softmax函数的一个性质
softmax函数存在一个性质：
$$softmax(X+c) = softmax(X)$$
这里$X$是向量，$c$是一个常数，证明过程如下：
$$\begin{aligned}
    softmax(X+c)_i =& \frac{e^{x_i +c}}{\sum_{k}e^{x_k +c}} \\
    =& \frac{e^{x_i}\cdot e^c}{\sum_{k}e^{x_k}\cdot e^c} \\
    =& \frac{e^{x_i}}{\sum_{k}e^{x_k}} \\
    =& softmax(X)_i
\end{aligned}$$
**在实际应用中，为了防止溢出，事先会把$x$减去最大值。**

参考地址：https://zhuanlan.zhihu.com/p/37740860