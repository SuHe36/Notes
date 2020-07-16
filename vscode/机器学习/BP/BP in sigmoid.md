## 1. sigmoid函数
下面是对sigmoid函数的定义：
$$\sigma(x) = sigmoid(x) = \frac{1}{1+ e^{-x}}$$

对其进行求导可得：
$$\begin{aligned}
    \sigma^{'}(x) =& \frac{e^{-x}}{(1+e^{-x})^2} \\
    =& \sigma(x)(1-\sigma(x))
\end{aligned}$$
## 2.结合交叉熵分析反向传播
一般sigmod(x)与交叉熵损失函数结合使用：
$$C= -(ylog\sigma(x) + (1-y)log(1-\sigma(x)))$$

这里的$y$是常量，是真实值，是目标值，下面求反向传播的推导过程：
$$\begin{aligned}
    \frac{\partial C}{\partial x} =& \frac{\partial C}{\partial \sigma(x)}\frac{\sigma(x)}{\partial x} \\
    =& -(\frac{y}{\sigma(x)} + \frac{1-y}{1-\sigma(x)}(-1))\sigma(x)(1-\sigma(x)) \\
    =& \sigma(x) - y
\end{aligned}$$


所以结合softmax函数来分析，：
- sigmoid函数 + 交叉熵
- softmax函数 + 交叉熵

参考地址：https://zhuanlan.zhihu.com/p/37773135