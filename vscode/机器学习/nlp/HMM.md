# Hidden Markov Models

HMM常用于处理一些序列问题，比如给句子$x_1,x_2,...,x_n$打上标签$y_1,y_2,...,y_n$，这通常就是叫做sequence labeling problem或者是tagging problem。

现在我们有一个训练集$(x^i,y^i),i=1,...,m$，其中每一个$x^i是一个句子序列x^i_1,x^i_2,...,x^i_n$，每一个$y^i是一个标注序列，y^i_1,y^i_2,...,y^i_n$。

其中x是输入,y是标签，我们的目标就是学习一个函数$f:x -> y$，将每个输入x，对应到它的标签y上。

一种定义函数f(x)的方法就是通过条件模型，我们首先定义一个条件概率：
$$p(y|x)$$
对于任意的一个输入x，模型的输出就是：
$$f(x) = \mathop{\arg \max}_{y \in \mathrm{y}} p(y|x) $$

而我们常用的则是一个**生成模型**，而不是直接使用一个条件分布$p(y|x)$，在生成模型中，我们使用联合概率：
$$p(x,y) $$
$p(x,y)$可以通过如下方式计算得到：
$$p(x,y) = p(y)p(x|y) $$
其中：
 - $p(y)$是对于标签y的一个先验概率
 - $p(x|y)$是在给与标签y的条件下，生成输入x的概率
  
**这个y的概率分布已知，我们可以称其为先验概率，而对于p(x|y)，它是在获得观察时间y之后得到的，所以我们可以称它为后验概率。**

由贝叶斯公式可得：
$$ p(y|x) = \frac{p(y)p(x|y)}{p(x)} $$

其中
$$p(x) = \sum_{y\in \mathrm{y}}p(x,y) = \sum_{y\in \mathrm{y}}p(y)p(x|y) $$

**所以对于一个数据集来说，可以把p(x)看做一个定值。**

所以我们可以将f(x)定义如下：
$$\begin{aligned}
    f(x) &= \mathop{\arg\max}_{y} p(y|x) \\
        &=\mathop{\arg\max}_{y}\frac{p(y)p(x|y)}{p(x)} \\
        &=\mathop{\arg\max}_{y}p(y)p(x|y)
\end{aligned}
$$

总结：

- 我们的任务就是学习一个函数f(x)，对于给与我们一个x，我们需要找到一个最可能的标签y=f(x)
- 我们需要使用训练集去评估p(y)和p(x|y),这个模型定义了一个生成模型：
  $$p(x,y) = p(y)p(x|y) $$
- 对于给与的一个测试样例x，我们预测它的标签：
- $$f(x) = \mathop{\arg\max}_{y \in \mathrm{y}}p(y)p(x|y) $$
给与输入x，找到它们的输入f(x)，往往被定义为一个解码问题。

## Generative Tagging Models
定义Generative Tagging Models：假设我们有一个有限的词集合$V$，和一个有限的标签集合$K$，定义$S$是sequence/tag-sequence的集合对$<x_1,..,x_n,y_1,...,y_n>$，其中$n \geq0,x_i \in V,并且y_i \in K$，那么一个生成模型$p$可以定义如下：
- 对于任意的$<x_1,...,x_n,y_1,...,y_n> \in S$
 $$ p(x_1,...,x_n,y_1,...,y_n) \geq 0$$
- 此外：
  $$\sum_{<x_1,...,x_n,y_1,...,y_n> \in S} p(x_1,...,x_n,y_1,...,y_n)=1 $$

  那么对于一个生成模型，函数f可以定义如下：
  $$f(x_1,...,x_n) = \mathop{\arg\max}_{y_1,...,y_n}p(x_1,...,x_n,y_1,...,y_n) $$

  那么接下来就有三个很重要的问题：
  - 我们如何定义一个生成模型$p(x_1,...,x_n,y_1,...,y_n)$
  - 我们如何从训练集中学习到模型参数
  - 对于任意的输入$x_1,...,x_n$，我们如何找到
  $$ \mathop{\arg\max}_{y_1,...,y_n}p(x_1,...,x_n,y_1,...,y_n)$$


  ## Trigram HMM
  下面我们以Trigram HMM为例，来整体介绍HMM算法的整个流程：
  