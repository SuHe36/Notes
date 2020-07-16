## tanh函数
tanh函数的定义如下：
$$\tau(x) = tanh(x) = \frac{e^x -e^{-x}}{e^x + e^{-x}}$$
求导过程如下：
$$\begin{aligned}
    \tau^{'}(x) =& \frac{4e^{-2x}}{(1+e^{-2x})^2} \\
    =& \frac{(1+e^{-2x})^2 - (1-e^{-2x})^2}{1+e^{-2x})^2} \\
    =& 1 - \frac{(1-e^{-2x})^2}{1+e^{-2x})^2} \\
    =& 1- \tau^{2}(x)
\end{aligned}$$