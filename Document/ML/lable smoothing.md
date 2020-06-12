#label smoothing

##1 背景介绍
在多分类训练任务中，输入样本经过神经网络计算，会得到当前样本对应于各个类别的置信度分值，这些分数会被softmax进行归一化处理，最终得到当前输入图片属于每个类别的概率。
$$ q_i = \frac{exp(z_i)}{\sum_{j=1}^{K}{exp(z_j)}}$$

之后在使用交叉熵函数来计算损失值：
$$ Loss = -\sum_{i=1}^{K}{p_ilogq_i}  $$
$$ p_i = 
\begin{cases}
    1 & \text{if(i==y)} \\
    0 & \text{if($i\neq y$)} 
\end{cases}$$
$$其中i表示多类中的某一类$$
**最终在训练网络时，最小化预测概率和标签真实概率的交叉熵，从而得到最优的概率预测分布。在此过程中，为了达到最好的拟合效果，最优的预测概率分布为：**
$$Z_i = 
\begin{cases}
  +\infty, & \text{if(i==y)} \\
  0,  & \text{$if(i\neq y)$}
\end{cases}$$
**也就是说，网络会驱使自身往正确标签和错误标签差值最大的方向学习，在训练数据不足以表征所有的样本特征的情况下，这就会导致网络过拟合。**

##2 label smoothing原理
- **label smoothing的提出就是为了解决上述问题。最早是在Inception v2中被提出，是一种正则化的策略。其通过"软化"传统的one-hot类型标签，使得在计算损失值时能够有效抑制过拟合现象**
- **label smoothing相当于减少真实样本标签的类别在计算损失函数时的权重，最终起到抑制过拟合的效果。**
  
###label smoothing将真实概率分布做了如下改变：
原有的概率为：
$$ p_i = 
\begin{cases}
    1 & \text{if(i==y)} \\
    0 & \text{if($i\neq y$)} 
\end{cases}$$
**修改后的概率为：**
$$ p_i = 
\begin{cases}
    1-\varepsilon & \text{if(i==y)} \\
    \frac{\varepsilon}{K-1} & \text{if($i\neq y$)} 
\end{cases}$$
$$其中K表示多分类的总类别数，\varepsilon是一个较小的超参数 $$
**其实更新后的分布就相当于往真实分布中加入了噪声，为了便于计算，该噪声服从简单的均与分布。**

###与之对应，label smoothing将交叉熵损失函数做了如下改变
原有的交叉熵损失函数为：
$$ Loss = -\sum_{i=1}^{K}{p_ilogq_i}  $$
**修改后的损失函数为：**
$$Loss_i = 
\begin{cases}
    (1-\varepsilon)*Loss, & if(i==y)\\
    \varepsilon *Loss, & if(i\neq y)
\end{cases}$$

###与之对应，label smoothing将最优的预测概率分布做了如下改变：
原有的最优预测概率如下：
$$Z_i = 
\begin{cases}
  +\infty, & \text{if(i==y)} \\
  0,  & \text{$if(i\neq y)$}
\end{cases}$$
**修改后的最优预测概率为：**
$$Z_i = 
\begin{cases}
    log\frac{(k-1)(1-\varepsilon)}{\varepsilon + \alpha}, & if(i==y) \\
    \alpha, & if(i\neq y)
\end{cases}$$
其中$\alpha$可以是任意实数，最终通过抑制正负样本输出差值，使得网络有更好的泛化能力。