# word2vec的总结

## 1. 传统的神经网络语言模型
传统的神经网络语言模型(NNLM)，一般都有三层，输入层(一般可以认为是一个N*V的矩阵，也就是最后训练想要得到的词向量矩阵)、隐藏层和输出层(softmax层)，但是这种模式存在一个问题，就是从隐藏层到softmax层的计算量过大，因为对于每一个词都要计算一个词表V大小的概率，然后再找概率最大的值，整个过程如下：
![](../figure/22.png)

这个神经网络包含四个层：输入层(Input)、投影层(Projection)、隐藏层(Hidden)和输出层(output).其中$W,U$分别为投影层和隐藏层，以及隐藏层和输出层之间的权值矩阵，$p,q$分别为隐藏层和输出层上的偏置向量。
当提到神经网络时，人们一般想的是如下的3层结构，但是本文为了和word2vec中的神经网络作对比，使用了如上的四层网络结构。
![](../figure/64.png)

神经网络词向量语言模型在word2vec出现之前就已经有了，他也是包含两种模型结构：$CBOW$与$Skip-gram$。
可以看到在输出层上需要进行softmax操作，但是当词表很大时，这一步将非常耗时，后续的word2vec也有对这一步的改进。



## 2. Word2Vec
word2vec的提出主要有以下两点改进：
 1. 第一个改进是：从输入层到隐藏层的映射，没有采取神经网络的线性变换再加激活函数的方式(也就是第一层是一个dense层)，而是直接使用简单的将所有输入的词向量加和取平均的方法。
 2. 第二个改进是：从隐藏层到输出的softmax层这里的计算量的改进。以前对于每个词都要计算词表V大小的概率分布，计算量大小是V，现在采用了哈夫曼树，计算量大小就变为了$log_2V$

word2vec的模型也是两种框架:CBOW和skip-gram，但是并没有像传统的神经网络语言模型一样采取DNN模型，它最先使用的是用霍夫曼树来代替隐藏层和输出层的神经元。
霍夫曼树中的叶子节点起到输出层神经元的作用，叶子节点的个数也就是词汇表的大小，而树中的内部节点就起到隐藏神经元的作用。

本文不再对CBOW和skip-gram的框架进行描述，主要讲基于这两种框架，word2vec做的两种改进：Hierarchical softmax和negative sample。
参考地址：https://blog.csdn.net/itplus/article/details/37998797

### 2.1 Hierarchical Softmax
Hierarchical softmax是word2vec中用于提高性能的一项关键技术，在引入之前，我们先介绍一些概念（为了构造哈夫曼树）：
1. $p^w$：从根节点出发到达$w$所对应的叶子节点的路径，路径中包含一个一个的节点；
2. $p^w_1,p^w_2,...,p^w_{l^w}$：路径$p^w$中的$l^w$个节点，其中$p^w_1$表示根节点，$p^w_{l^w}$表示词$w$对应的节点；
3. $l^w$：路径$p^w$中包含的节点的个数。
4. $d_2^w,d_3^w,...,d_{l^w}^w \in {0,1}$：词$w$的$Huffman$编码，它由$l^w-1$位编码构成，$d^w_j$表示路径$p^w$中第j个节点对应的编码（根节点不对应编码）。
5. $\theta_1^w, \theta_2^w,...,\theta_{l^w-1}^w \in \mathbb{R}^m$：路径$p^w$中非叶子节点对应的向量，$\theta_j^w$表示路径$p^w$中第j个非叶子节点对应的向量，$\theta_j^w$可以理解为是在Huffman树中确定往左往右的参数向量。

#### 2.1.1 CBOW模型
如下图，是一个预测词$w=$”足球“的例子，这里使用的是CBOW框架，也就是用周围词预测中心词
![](../figure/65.png)

我们可以发现，从根节点到达"足球"这个叶子节点，中间共经历了四个分支，而每一个分支都可以视为一个二分类，那么我们就可以定义：对一个节点进行分类时，分到左边就是负类，分到右边就是正类，那么从根节点到叶子节点的路径对应的分类过程，其实就是一个Huffman编码。

在分类时，使用的是逻辑回归，所以，一个节点被分为正类的概率为：
$$\sigma(x_w^T \theta) = \frac{1}{1 + e^{-x_w^T\theta}} $$
那么，相应的被分为负类的概率就是：
$$1- \sigma(x_w^T \theta)$$
其中的$\theta$就是上面所说的非叶子节点所对应的那些向量$\theta_i^w$， $x_w^T$在cbow中就是周围词的词向量进行相加得到的，即$x_w = \sum_{i=1}^{2c}v(Context(w_i)) \in \mathbb{R}^m$。

对于上面那个从根节点出发到达叶子节点"足球"的路径，共经历了四次二分类，每次分类的结果就是：
1. 第一次（分到左边）：$p(d^w_2|x_w, \theta_1^w) = 1-\sigma(x_w^T\theta_1^w)$;
2. 第二次（分到右边）：$p(d^w_3|x_w, \theta_2^w) = \sigma(x_w^T\theta_2^w)$;
3. 第三次（分到右边）：$p(d^w_4|x_w, \theta_3^w) = \sigma(x_w^T\theta_3^w)$;
2. 第四次（分到左边）：$p(d^w_5|x_w, \theta_4^w) = 1-\sigma(x_w^T\theta_4^w)$;

那么我们最后要求的$p(足球|Context(足球))$，他就是由这四个概率值连乘后得到的：
$$ p(足球|Context(足球)) = \prod_{j=2}^{5}p(d_j^w|x_w,\theta^w_{j-1}) $$
总结一下：**对于词典D中的任意词$w$，Huffman树中必存在一条从根节点出发到词$w$对应的叶子节点的路径，并且这条路径是唯一的，在这条路径上会存在若干次二分类，而每次二分类就会产生一个概率值，把这些概率值连乘起来，就是所需的$p(w|Context(w))$**
所以，整个cbow的由周围词预测中心词的概率可以写成如下这种方式：
$$p(w|Context(w)) = \prod_{j=2}^{l^w} p(d_j^w|x_w, \theta_{j-1}^w) $$
其中
$$p(d_j^w|x_w,\theta_{j-1}^w) = 
\begin{cases}
    \sigma(x_w^T\theta^w_{j-1}), \qquad &d_j^w=0;\\
    1 - \sigma(x_w^T\theta^w_{j-1}), \qquad &d_j^w=1;
\end{cases}
$$
写成整体表达式就是：
$$ p(d_j^w|x_w,\theta_{j-1}^w) = [\sigma(x_w^T\theta_{j-1}^w)]^{1-d_j^w} \cdot [1-\sigma(x_w^T\theta_{j-1}^w)]^{d_j^w}$$


而我们整个网络的目标是让词典D中的所有词出现的概率最高，也就是神经网络语言模型的目标函数是：
$$L = \sum_{w\in D} log p(w|Context(w))  $$

所以把我们上面用cbow求得的一个词的概率，代入目标函数可得：
$$\begin{aligned}
    L &= \sum_{w\in D}log \prod_{j=2}^{l^w}p(d_j^w|x_w, \theta_{j-1}^w) \\
      &= \sum_{w\in D}log \prod_{j=2}^{l^w}[\sigma(x_w^T\theta_{j-1}^w)]^{1-d_j^w} \cdot [1-\sigma(x_w^T\theta_{j-1}^w)]^{d_j^w} \\
      &= \sum_{w\in D}\sum_{j=2}^{l^w} {(1-d_j^w)\cdot log[\sigma(x_w^T\theta_{j-1}^w)]  + d_j^w\cdot log[1-\sigma(x_w^T\theta_{j-1}^w)]}
\end{aligned}$$
这个式子就是CBOW的目标函数，接下来我们要讨论的就是如何优化它。

为了下面的梯度推导方便，我们可以将上式的双重求和符号中的内容统一表示成$L(w,j)$，即：
$$L(w,j) = (1-d_j^w)\cdot log[\sigma(x_w^T\theta_{j-1}^w)]  + d_j^w\cdot log[1-\sigma(x_w^T\theta_{j-1}^w)]$$

word2vec中采用的是梯度上升法，因为是要求出现的概率越大越好。首先，我们求一下$L(w,j)$关于$\theta_{j-1}^w$的梯度计算：
$$\begin{aligned}
    \frac{\partial L(w,j)}{\partial \theta_{j-1}^w} &= \frac{\partial}{\partial \theta_{j-1}^w} {(1-d_j^w)\cdot log[\sigma(x_w^T\theta_{j-1}^w)]  + d_j^w\cdot log[1-\sigma(x_w^T\theta_{j-1}^w)]}  \\
    &= (1-d_j^w)[1-\sigma(x_w^T\theta_{j-1}^w)]x_w - d_j^w \sigma(x_w^T\theta_{j-1}^w)x_w \\
    &= [1-d_j^w - \sigma(x_w^T\theta^w_{j-1})]x_w 
\end{aligned}$$

因此，$\theta_{j-1}^w$的更新公式可以写成：
$$\theta_{j-1}^w := \theta_{j-1}^w + \eta[1-d_j^w - \sigma(x_w^T\theta_{j-1}^w)]x_w$$
其中$\eta$表示学习率。
同理可以求得对$w_w$的梯度：
$$\frac{\partial L(w,j)}{\partial x_w}  = [1-d_j^w -\sigma(x_w^T \theta^w_{j-1})]\theta_{j-1}^w$$


我们的最终的目标是要求的词典中的每个词的词向量，而这里的$x_w$是Context(w)中的各个周围词词向量的累加，那么我们如何利用$\frac{\partial L(w,j)}{\partial x_w}$来对周围词$v(\hat{w}), \hat{w} \in Context(w)$进行更新呢？ 
word2vec的做法很简单，就是对每一个周围词的词向量进行更新，注意，我们上面求得$\frac{\partial L(w,j)}{\partial x_w}$只是一次二分类时产生的对$x_w$的梯度，最终对周围词进行更新时，是要用到所有二分类时产生的梯度来进行更新,即：
$$v(\hat{w}) := v(\hat{w}) + \eta \sum_{j=2}^{l^w} \frac{\partial L(w,j)}{\partial x_w},\qquad \hat{w} \in Context(w) $$

#### 2.1.2 Skip-Gram模型
Skip-gram模型与CBOW的模型结构大同小异，具体结构如下图所示：
![](../figure/66.png)

Skip-gram的目标函数为：
$$L = \sum_{w\in D} log p(Context(w)|w)  $$
Skip-gram中利用中心词w来预测周围词，所以skip-gram中的条件概率p(Context(w)|w)为：
$$ p(Context(w)|w) = \prod_{u\in Context(w)} p(u|w)$$

按照上面的Hierarchical Softmax的思想，可以将其写为：
$$p(u|w) = \prod_{j=2}^{l^u} p(d_j^u|v(w),\theta_{j-1}^u) $$

其中：
$$p(d_j^u|v(w), \theta_{j-1}^u) = [\sigma(v(w)^T\theta_{j-1}^u)]^{1-d_j^u} \cdot [1-\sigma(v(w)^T\theta_{j-1}^u)]^{d_j^u} $$

然后代入目标函数【同CBOW的目标函数基本一致】后可得:
$$\begin{aligned}
    L =& \sum_{w\in D} log \prod_{w\in Content(w)} \prod_{j=2}^{l^u}{[\sigma(v(w)^T\theta_{j-1}^u)]^{1-d_j^u} \cdot [1-\sigma(v(w)^T\theta_{j-1}^u)]^{d_j^u}} \\
      =& \sum_{w\in D}\sum_{u \in Context(w)} \sum_{j=2}^{l^u}{(1-d_j^u)\cdot log[\sigma(v(w)^T\theta^u_{j-1})] + d_j^u \cdot log[1-\sigma(v(w)^T\theta_{j-1}^u)]} 
\end{aligned}$$

同样的为了下面推导方便，我们将三重求和符号下的内容记为$L(w,u,j)$，即：
$$L(w,u,j) =(1-d_j^u)\cdot log[\sigma(v(w)^T\theta^u_{j-1})] + d_j^u \cdot log[1-\sigma(v(w)^T\theta_{j-1}^u)] $$

Skip-gram同样利用梯度上升法求解。我们可以先求的$L(w,u,j)$对于$\theta_{j-1}^u$的梯度计算：
$$\begin{aligned}
    \frac{\partial L(w,u,j)}{\partial \theta_{j-1}^u} =& \frac{\partial}{\partial \theta_{j-1}^u} (1-d_j^u)\cdot log[\sigma(v(w)^T\theta^u_{j-1})] + d_j^u \cdot log[1-\sigma(v(w)^T\theta_{j-1}^u)]\\
    =& [1-d_j^u -\sigma(v(w)^T\theta_{j-1}^u)]v(w).
\end{aligned}$$

因此,$\theta_{j-1}^u$的更新公式可以写成：
$$\theta_{j-1}^u := \theta_{j-1}^u + \eta[1-d_j^u -\sigma(v(w)^T\theta_{j-1}^u)]v(w) $$

同理可以计算出$L(w,u,j)$对于$v(w)$的梯度：
$$\frac{\partial L(w,u,j)}{\partial v(w)} = [1-d_j^u -\sigma(v(w)^T\theta_{j-1}^u)]\theta_{j-1}^u $$

所以$v(w)$的更新公式可以写成：
$$v(w) := v(w) + \eta \sum_{u \in Context(w)}\sum_{j=2}^{l^u} \frac{\partial L(w,u,j)}{\partial v(w)}$$

事实上，在源码中并不是等所有的context(w)中的词都处理完才更新$v(w)$，而是，每处理完Context(w)中的一个词$u$，就及时刷新一次$v(w)$。

### 2.2 Negative Sample

本文将继续介绍基于Negative Sampling的CBOW和Skip-gram模型。
Negative sample是用来提高训练速度并改善词向量的质量。与Hierarchical softmax相比，neg不再使用复杂的Huffman树，而是利用相对简单的**随机负采样**，能够大幅提高性能，因此可以作为Hierarchical softmax的一种替代。
我们还是结合CBOW和Skip-gram两种模型来对neg进行细致介绍。
#### 2.2.1 CBOW模型
在CBOW模型中，已知词$w$的上下文$Context(w)$，需要预测$w$，因此，对于给定的Context(w)，词w就是一个正样本，其他词就是一个负样本了，负样本那么多，该如何选取呢？本文的后续在探讨这个问题，我们先假设有一个选好了的负样本集，那么我们要如何进行训练。
假设$NEG(w) \neq 0$就是我们选好了的关于词$w$的负样本集，并且对$\forall \hat{w} \in D$，定义：
$$L^w(\hat{w}) = 
\begin{cases}
    1, \quad &\hat{w} = w; \\
    0, \quad &\hat{w} \neq w;
\end{cases}$$
即正样本的标签为1，负样本的标签为0。

对于一个给定的正样本$Context(w),w$，我们希望最大化：
$$g(w) = \prod_{u \in NEG(w)} p(u|Context(w)) $$
其中：
$$p(u|Context(w)) = 
\begin{cases}
    \sigma(x_w^T \theta^u), \quad &L^w(u) = 1; \\
    1 - \sigma(x_w^T \theta^u), \quad &L^w(u) = 0;
\end{cases}$$

或者写成整体的：
$$p(u|Context(w)) = [\sigma(x_w^T\theta^u)]^{L^w(u)} \cdot [1-\sigma(x_w^T\theta^u)]^{1-L^w(u)}$$
这里的$x_w$仍然表示Context(w)中各词的词向量之和，而$\theta^u \in \mathbb{R}^m$表示词$u$对应的一个向量，为待训练的参数。

我们把$p(u|Context(w))$代入$g(w)$的表达式，可得：
$$g(w) = \sigma(x_w^T\theta^w) \prod_{u \in NEG(w) \bigcup u \neq w}[1-\sigma(x_w^T\theta^u)]$$
其中$\sigma(x_w^T\theta^w)$表示上下文为Context(w)时，预测中心词为w的概率，而$\sigma(x_w^T\theta^u)$则表示当上下文为context(w)是，预测中心词为u的概率。
从形式上看，最大化$g(w)$，相当于最大化$\sigma(x_w^T\theta^w)$，同时最小化所有的$\sigma(x_w^T\theta^u), u \in NEG(w)$。
这正如我们所希望的那样，**增大正样本的概率，同时降低负样本的概率。**于是，对于一个给定的语料库$C$，函数：
$$G = \prod_{w\in C}g(w)$$

函数$G$就可以作为整体优化目标。当然，为了计算方便，我们可以对$G$取对数，最终的目标函数就是：
$$\begin{aligned}
    L =& log G = log\prod_{w\in C}g(w) \\
      =& \sum_{w \in C}log g(w) \\
      =& \sum_{w \in C} \prod_{w \in NEG(w)}{[\sigma(x_w^T\theta^u)^{L^w(u)}] \cdot [1-\sigma(x_w^T\theta^u)]^{1-L^w(u)} } \\
      =& \sum_{w\in C} \prod_{w\in NEG(w)}{L^w(u)\cdot log[\sigma(x_w^T\theta^u)] + [1-L^w(u)]\cdot log[1-\sigma(x_w^T\theta^u)]}
\end{aligned}$$

把w区分是正例和负例，上式也可以写成：
$$\begin{aligned}
    L = \sum_{w\in C}{log[\sigma(x_w^T\theta^w)] + \sum_{w \in NEG(w) \bigcup u \neq w}log[\sigma(-x_w^T\theta^u)] } (利用了等式1-\sigma(x)=\sigma(-x))
\end{aligned}$$

同理，我们使用$L(w,u)$简记上式花括号里面的内容，也就是：
$$L(w,u) = L^w(u)\cdot log[\sigma(x_w^T\theta^u)] + [1-L^w(u)]\cdot log[1-\sigma(x_w^T\theta^u)]$$

这里也是使用梯度上升法来进行求解的，先求$\frac{\partial L(w,u)}{\partial \theta^u}$：
$$\frac{\partial L(w,u)}{\partial \theta^u} = [L^w(u) - \sigma(x_w^T\theta^u)]x_w$$
于是$\theta^u$的更新公式为：
$$\theta^u := \theta^u + \eta [L^w(u) - \sigma(x_w^T\theta^u)]x_w $$
同理，求得$\frac{\partial L(w,u)}{\partial x_w}$的：
$$\frac{\partial L(w,u)}{\partial x_w} = [L^w(u) - \sigma(x_w^T\theta^u)]\theta^u $$

这里的$x_w$还是Context(w)的词向量之和，与前面Hierarchical Softmax用所有的$L(w,j)$对$v(\hat{w})$中心词的词向量进行更新相同，这里用所有的$L(w,u)$，包括正确的中心词和错误的负类的$L(w,u)$来对中心词$v(\hat{w})$进行更新：
$$v(\hat{w}) := v(\hat{w}) + \eta \sum_{w \in NEG(w)} \frac{\partial L(w,u)}{\partial x_w}, \quad \hat{w} \in Context(w) $$

#### 2.2.2 Skip-gram模型
同理，negative sample下的skip-gram的目标函数由CBOW的：
$$G = \prod_{w \in C} g(w)$$
改写为：
$$G = \prod_{w \in C} \prod_{u \in Context(w)} g(u) $$
这里的$\prod_{u \in Context(w)} g(u)$表示对于一个给定的样本(w, Context(w))，我们希望最大化的量就是这个。
g(u)的定义如下：
$$g(u) = \prod_{z \in NEG(u)} p(z|w)$$
其中$NEG(u)$表示处理词$u$时生成的负样本子集和正例$u$，条件概率如下：

$$p(z|w) = 
\begin{cases}
    \sigma(v(w)^T \theta^z), \quad &L^u(z) = 1; \\
    1 - \sigma(v(w)^T \theta^z), \quad &L^u(z) = 0;
\end{cases}$$

写成整体的表达式：
$$p(z|w) = [\sigma(v(w)^T\theta^z)]^{L^u(z)} \cdot [1-\sigma(v(w)^T\theta^z)]^{1-L^u(z)} $$

同样的，代入目标函数，以及对目标函数取对数可得：
$$\begin{aligned}
    L =& log G \\
      =& \sum_{w\in C} \sum_{u \in Context(w)} \sum_{z \in NEG(u)} {L^u(z)\cdot log[\sigma(v(w)^T\theta^z)] + [1-L^u(z)]\cdot log[1-\sigma(v(w)^T\theta^z)]}
\end{aligned}$$

**这里的负采样需要说一下，按照一般的理解是，CBOW需要预测中心词，因此对中心词进行负采样。那么skip-gram中是预测周围词，那么也应该对周围词进行负采样。**
**word2vec中，对于skip-gram来说，他并不是对于Context(w)中的每一个周围词进行负采样，而是针对$w$进行了|Context(w)|词负采样。**
**这样做的具体原因，目前的解释不是很清楚。**
后续的推导就不写了。

#### 2.2.3 负采样算法
词典$D$中的词在语料$C$中出现的次数有高有低，那么对于那些高频词，被选为负样本的概率就应该大一些，反之，对于那些低频词，其被选中的概率就应该比较小，这是我们队负采样算法的一个基本要求。
下面具体描述一下，具体的负采样算法：
- 设词典$D$中的每一个词$w$对应一个线段$l(w)$，长度为：
- $$len(w) = \frac{counter(w)}{\sum_{u \in D}counter(u)} $$
- 其中$counter(w)$表示词$w$在语料中出现的次数，显然词频越高，线段也就越长。

具体到word2vec中时，计算线段长度时，他对其做了幂操作：
$$len(w) = \frac{[counter(w)]^{\frac{3}{4}}}{\sum_{w\in D}[counter(u)]^{\frac{3}{4}}}$$

那么我们可以构造一个线段，这个线段总共由N条线段组成的，每一段的长度也就是$len(i)$，那么总长为$L = \sum_{i =1}^{N}len(i)$。
我们可以将这个线段等分成$M$份，其中$M >> N$，那么数值$j\in [1,M-1]$就会落在某个线段内，具体如下所示：
![](../figure/67.png)

所以，我们最后会得到一个数值$j\in [1,M-1]$与$len(w_i), i\in[1,N]$的映射关系。
那么采样就很简单了：每次生成一个$[1,M-1]$间的随机整数，然后对应到的$w_i$就是负样本，如果碰巧$w_i$是正例自身，那么就跳过去即可。

### 3. cbow和skip-gram的区别
CBOW使用的是周围词预测中心词，Skip-Gram使用的是中心词预测周围词之外。
并且上文中的$x_w$和$\theta^w$，在有的文章中，也被描述为词$w$的输入词向量$v(w)$和输出词向量$u(w)$，或者叫做中心词词向量和背景词词向量。

所以Skip-gram模型是用中心词来预测周围词，所以最后选用中心词的词向量作为词的表征特征。
而CBOW是用周围词预测中心词，所以选用的是背景词的词向量作为词的表征特征。
其实都是把输入词向量$x_w$作为最后的词的表征特征，中间二分类引入的参数$\theta$，有些人也把他叫做词的背景词词向量（在skip-gram里）或中心词词向量（在cbow里）。