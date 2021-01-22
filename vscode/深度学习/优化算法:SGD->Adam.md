几个最有效的博客

https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92ml-note-sgd-momentum-adagrad-adam-optimizer-f20568c968db

https://medium.com/datadriveninvestor/overview-of-different-optimizers-for-neural-networks-e0ed119440c3

https://zhuanlan.zhihu.com/p/32230623

https://blog.csdn.net/q295684174/article/details/79130666

公众号比较全面的总结：
https://mp.weixin.qq.com/s/S42EafIC7q8uqMm9Urb-Kg

这里写一下简单的总结。


优化算法的整个流程如下：
- 1, 计算目标函数关于当前参数的梯度:$g_t = \nabla f(\theta_t)$
- 2,根据历史梯度计算一阶动量和二阶动量:
$$m_t =\phi(g_1,g_2,...,g_t); \quad V_t = \psi (g_1,g_2,...,g_t) $$
- 3,计算当前时刻的下降梯度： $\eta_t = \alpha \cdot m_t/\sqrt{V_t}$，其实可以写成：$\eta_t = (\alpha/\sqrt{V_t})\cdot m_t$
- 4,根据下降梯度进行更新：$\theta_{t+1}=\theta_t - \eta_t$

其实各种优化算法对于上面的步骤3，4都是一致的，主要的差别体现在步骤1和步骤2上。



SGD：

- 就是最基本的梯度下降，但它最大的缺点就是下降速度慢，而且可能会在沟渠的两边持续震荡，停留在一个局部最优点。

$$\theta_{t} = \theta_{t-1} - \alpha \nabla J(\theta;x,y) $$


SGD-M: SGD with Momentum：
- 在SGD的基础上，引入了一阶动量，一阶动量也就是指各个时刻梯度方向的移动平均值。
也就是说当前t时刻的梯度方向，不仅由当前点的梯度方向决定，而且由此前累积的下降方向决定的。

- 这里相当于在SGD的基础上对梯度的大小进行了改变。

$$\begin{aligned}
    v_t &= \beta v_{t-1} + \alpha \nabla J(\theta;x,y) \\
    \theta_{t} & = \theta_{t-1} - v_t
\end{aligned}$$


NAG:Nesterov accelerated gradient
- NAG是在SGD和SGD-M的基础上的进一步改进，**它的改进点在步骤1。**
- SGD有一个问题就是在局部最优的沟壑里震荡。想象一下你走到一个盆地，四周都是略高的小山，你觉得没有下坡的方向，那就只能待在这里。可是如果你爬上高地，就会发现外面的世界还是很广阔的。因此，我们不能停留在当前位置去观察未来的方向，而是要向前一步，多看一步，看远一些、
- 它在t时刻，不是按照当前梯度方向走的，而是跟着累积动量走了一步，走了之后再看下一步该怎么走。

具体的参数更新方式如下：
$$\begin{aligned}
    v_t &= \beta v_{t-1} + \alpha \nabla J(\theta_{t-1} - \beta v_{t-1}) \\
    \theta_{t} &= \theta_{t-1} - v_{t} \\
    其中&\theta_{t-1} - \beta v_{t-1}就是向前看得一步
\end{aligned}$$

AdaGrad:
- 这里引入了二阶动量，二阶动量就是指，从0时刻开始到t时刻，所有梯度值的平方和。
- AdaGrad就是会依照梯度去调整学习率lr的优化器，学习率会除以二阶动量的根植，一般为了避免分母为0，会加上一个小的平滑项。
- 这样就可以实现，当前期梯度较小时，二阶动量较小，能够放大学习率；当后期梯度较大时，n较大，能够约束学习率。但是仍然存在一个问题，就是分母上梯度平方的累加会越来越大，会使学习率接近于0，训练便会结束。

$$\begin{aligned}
    \theta_{t} &= \theta_{t-1} - \frac{\alpha}{\sqrt{G_{t-1}+\varepsilon}} \cdot g_t \\
    其中 G_t是一个对角矩阵，其&对角元素是参数\theta截止t时刻的所有梯度的平方和。\\
    也就是上面的二阶动量V_{t-1} &
\end{aligned}$$

AdaDelta/RMSprop:
- 由于AdaGrad的在后期梯度平方和过大时会使学习率接近于0，所以考虑一个改进二阶动量计算方法的策略。
- AdaDelta的改进就是不累积全部历史梯度，而只关注过去一段时间窗口的下降梯度。在AdaDelat中，我们不需要设置具体的学习率，因为它把前一时刻的梯度的均值和当前时刻的梯度的均值的比值作为了学习率。
- RMSprop的改进则是窗口滑动的加权平均来解决梯度的消失问题。【这个的具体公式怎么起作用的，我也没看明白。】

AdaDelta和RMSprop的具体公式可以看第二个和第四个的参考链接。

AdaDelta的公式如下：
$$\begin{aligned}
    \Delta \theta_t &= -\frac{RMS[\Delta \theta]_{t-1}}{RMS[g_t]} \cdot g_t \\
    \theta_{t+1} &= \theta_{t} +  \Delta \theta_t \\
    其中&RMS是均方根
\end{aligned}$$


RMSprop的公式如下:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{(1-\beta)g^2_{t-1} + \beta g_t + \varepsilon}} \cdot g_t$$


Adam:
- Adam就是前述所有方法的集大成者，Adam就是把一阶动量和二阶动量都用了起来。
- 也就是说，Adam会更新学习率，也会更新梯度大小。

$$\begin{aligned}
    m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t \\
    v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
\end{aligned}$$
$m_t和v_t$分别是一阶动量和二阶动量 .

对$m_t和v_t$做偏离校正。

$$\begin{aligned}
    \hat{m_t} &= \frac{m_t}{1-\beta_1^t} \\
    \hat{v_t} &= \frac{v_t}{1-\beta_2^t}
\end{aligned}$$

最终的参数更新如下式：
$$\theta_{t+1} = \theta_{t} - \frac{\alpha \hat{m_t}}{\sqrt{\hat{v_t}+\varepsilon}}$$

Nadam:
- 最后就是Nadam，他其实就是NAG和Adam的合成。也就是它与Adam相比，就是在第1步的时候变成了：
$$g_t = \nabla f(\theta_t - \alpha\cdot m_{t-1}/\sqrt{V_t})$$
