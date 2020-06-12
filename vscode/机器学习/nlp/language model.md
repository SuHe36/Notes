# Language Model

## 什么是语言模型
定义：一个语言模型包含一个有限的词表集合$V$，和一个函数$p(x_1,x_2,...,x_n)$，表示句子$x_1,x_2,...,x_n$出现的概率，总结如下：
  1. 对于任意的$<x_1,x_2,...,x_n> \in V$，都有$p(x_1,x_2,...,x_n) \geq 0$
  2. 此外 $\sum_{<x_1,x_2,...,x_n> \in V} p(x_1,x_2,...,x_n)=1$

一个最简单的从一些训练语料中学习语言模型的例子：定义$c(x_1,x_2,...,x_n)$是句子$x_1,x_2,...,x_n$出现在训练语料中的次数，$N$是在训练语料中的所有句子，那么我们可以定义：
$$p(x_1,x_2,...,x_n) = \frac{c(x_1,x_2,...,x_n)}{N}$$
显然这有一个很严重的问题，对于那些没有在训练语料中出现的语句，直接将它的概率值赋为了0.
接下来我们主要讨论对于那些没有出现在训练语料中的句子，我们如何计算与评估它们的概率值。

## 马尔可夫模型
假设我们有一个有限的词表集合$V$，我们的目标是，对于任一的一个句子$x_1,x_2,...,x_n；n \geq 1并且x_i \in V$，计算出他们出现的概率值：
$$P(X_1=x_1,X_2=x_2,...,X_n=x_n) $$
那么总共可能有$|V|^n$个句子组合。
在first order的马尔可夫过程中，我们假设每个词的出现仅和它前面的第一个词有关，那么我们可以得到：
$$
\begin{aligned}
  &P(X_1= x_1,X_2=x_2,...,X_n=x_n)  \\
  =& P(X_1=x_1)\prod_{i=2}^{n} P(X_i=x_1|X_1=x_1,...,X_{i-1}=x_{i-1}) \\
  =& P(X_1=x_1)\prod_{i=2}^{n} P(X_i=x_i|X_{i-1}=x_{i-1})
\end{aligned}
$$
也就是在first order的马尔可夫过程中，我们假设：
$$ 
\begin{aligned}
&P(X_i=x_i|X_1=x_1,...,X_{i-1}=x_{i-1}) \\
&P(X_i=x_i|X_{i-1}=x_{i-1})

\end{aligned}
$$
那么在second order的马尔可夫过程中，我们假设每个词仅和它前面的两个词有关，那么我们可以得到：
$$ 
\begin{aligned}
&P(X_i=x_i|X_1=x_1,...,X_{i-1}=x_{i-1}) \\
&P(X_i=x_i|X_{i-2}=x_{i-2},X_{i-1}=x_{i-1})

\end{aligned}
$$

应用second order的马尔可夫过程，我们可以得到second order的马尔可夫模型：

$$
\begin{aligned}
  &P(X_1= x_1,X_2=x_2,...,X_n=x_n)  \\
  =& \prod_{i=1}^{n} P(X_i=x_i|X_{i-2}=x_{i-2},X_{i-1}=x_{i-1})
\end{aligned}
$$
我们可以假设$x_0=x_{-1}=*$，其实就是句子起始符的另种定义，并且定义$x_n=STOP$，表示句子停止符。
我们可以定义：
$$q(x_i|x_{i-2},x_{i-1}) = P(X_i=x_i|X_{i-2}=x_{i-2},X_{i-1}=x_{i-1}) $$
其中$q(w|u,v)$可以视为是任意的$(u,v,w)$的模型的参数，后续我们会介绍如何从训练语料中得到$q(w|u,v)$.$q(w|u,v)$也可以看做是在上文词$u,v$的限制下，$w$词的分布情况。

那么我们可以得到定义，对于任意的句子$x_1,x_2,...,x_n$，他们的概率计算方式为：
$$p(x_1,...,x_n)=\prod_{i=1}^{n} q(x_i|x_{i-2},x_{i-1})$$

那么对于任意的三元组$u,v,w$，都有：
$$q(w|u,v) \geq 0 $$

并且对于任意的二元组$u,v$，我们都有：
$$\sum_{w\in V \cup {STOP}} q(w|u,v)=1 $$
那么对于整个语言模型大概有$|V|^3$个参数。

## 最大似然估计
首先定义$c(u,v,w)$是三元组$(u,v,w)$在训练语料中出现的次数，那么$c(u,v)$就是二元组$(u,v)$在语料中出现的次数，那么对于任意的$u,v,w$，我们可以定义：
$$ q(w|u,v) = \frac{c(u,v,w)}{c(u,v)} $$
但是用这种方法来估计模型的参数，存在着一些问题：

1. 由于在语料中一些三元组出现的次数为0，那么它们的$q(w|u,v)$就被赋值为0，这样直接把那些没有在训练语料中出现的三元组赋值为0，显然是不合理的。
2. 分母上的值$c(u,v)$为0时，无法处理这种情况

## 平滑估计
对于计算
$$ q(w|u,v) = \frac{c(u,v,w)}{c(u,v)} $$
其分子分母可能为0的问题，人们提出了平滑估计(smoothed estinamtion)，主要用来处理稀疏数据的问题。

方法大致可以分为两种：linear interpolation和discount methods。

### linear interpolation
首先，我们定义traigram、bigram、unigram的最大似然估计如下：
$$
\begin{aligned}
&q_{ML}(w|u,v) = \frac{c(u,v,w)}{c(u,v)} \\
&q_{ML}(w|v) = \frac{c(v,w)}{c(v)} \\
&q_{ML}(w) = \frac{c(w)}{c()}


\end{aligned}
$$
其中$c()$是在训练语料中的所有词的数量。
我们可以得知unigram 的似然估计永远不可能为0，并且永远大于0，但是unigram似然估计完全忽略了前文的词。与之相反，trigram似然估计则充分利用了前文的信息，bigram似然估计则介于两者之间。

所以linear interpolation就是联合使用这三种估计，通过定义trigram model如下：
$$q(w|u,v) = \lambda_{1} \times q_{ML}(w|u,v) + \lambda_2 \times q_{ML}(w|v) + \lambda_3 \times q_{ML}(w) $$
其中$\lambda_1 \geq 0、\lambda_2 \geq 0、\lambda_3 \geq 0$，并且$\lambda_1 + \lambda_2 + \lambda_3 = 1$,其中 $\lambda_1 、\lambda_2、\lambda_3$可以视为超参，可以在验证集上使得其所有概率和最大时取得的值。

对这种方法的一个延伸，叫做bucketing方法，定义计算$\lambda_1 、\lambda_2、\lambda_3$的方法如下：
$$
\begin{aligned}
 &\lambda_1 = \frac{c(u,v)}{c(u,v)+ \gamma} \\
 &\lambda_2 = (1-\lambda_1) \times \frac{c(v)}{c(v)+\gamma} \\
 &\lambda_3 = 1-\lambda_1 -\lambda_2

\end{aligned}
$$
这时$\gamma$是这种方法中模型的唯一的一个参数。可以看到在这种方法中，当$c(u,v)$变大时，$\lambda_1$也会变大，同时$c(v)$变大时，$\lambda_2$也会变大。并且，如果$c(u,v)=0，那么\lambda_1=0，当c(v)=0，那么\lambda_2=0。$

### Discount Methods
discount methods经常被用在实际应用中.
首先第一步我们定义对于任意的二元组$c(v,w)，其中c(v,w) \gt 0$,：
$$c^{*}(v,w) = c(v,w) -\beta $$
其中$\beta$的取值范围是0到1，一般将$\beta$设置为0.5；
那么我们就定义:
$$q(w|v) = \frac{c^{*}(v,w)}{c(v)}$$

那么对于任意的一个词$v$，我们都有一个定义遗失概率值，定义如下：
$$\alpha(v) = 1 - \sum_{w:c(v,w) \gt 0} \frac{c^{*}(v,w)}{c(v)} $$

那么对于任意的一个词$v$，我们可以将其分为两个集合:
$$
\begin{aligned}
   & A(v) = {w:c(v,w) \gt 0} \\
   & B(v) = {w:c(v,w)=0}
\end{aligned}
$$
然后对于他们的似然估计概率值计算方法如下：
$$ q_D(w|v) = 
\begin{cases}
  &  \frac{c^*(v,w)}{c(v)} &If  \quad w \in A(v) \\
 & \alpha(v) \times \frac{q_{ML}(w)}{\sum_{w\in B(v)}q_{ML}(w)} &If \quad w \in B(v) 
\end{cases}
$$

也就是说，我们将那些遗失的概率值分配给了那些没有在训练集中出现的$c(v,w)$