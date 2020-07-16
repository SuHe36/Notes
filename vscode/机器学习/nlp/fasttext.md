#! https://zhuanlan.zhihu.com/p/160932151
## FastText
再聊fasttext之前，我们先来聊一下word2vec，因为fasttext就是基于word2vec做的一些改进。
### word2vec
下面以skipgram为例介绍word2vec中的计算方式。
给与一个语料库$w_1,w_2,...,w_T$，skipgram模型的训练目标是最大化下面的极大似然函数：
$$\sum_{t=1}^T\sum_{c\in C_t}logp(w_c | w_t)$$
其中$C_t$是在词$w_t$周围的词。

原始的NNLM模型求$p(w_c|w_t)$，是对词表中的所有词进行softmax概率预测，即:
$$p(w_c|w_t) = \frac{e^{s(w_t,w_c)}}{\sum_{j=1}^{W} e^{s(w_t,j)}}$$
其中$s()$函数可以理解为NNLM中的输出层等。
显然这种方式是对词表中的所有词都预测了一个概率值，十分耗时，而我们却是只需要对一个词预测。所以，word2vec的方法采用了Hierarchital Softmax的方法，通过采用哈夫曼树的方法，大大增加了预测速度（由$O(N)$的时间复杂度降到了$O(logN)$）.
在word2vec中，对于一个选中的周围词$w_c$，采用二分类逻辑损失函数，以及采取Negative Sample，具体的似然函数如下：
$$log(1+e^{-s(w_t,w_c)}) + \sum_{n \in N_{t,c}}log(1+e^{s(w_t,n)}) $$
其中$N_{t,c}$是采样的负例集合。
我们用函数$l$来表示$x -> log(1+e^{-x})$，我们可以把整个训练集的训练目标写成如下形式：
$$\sum_{t=1}^{T}\sum_{c\in C_t}{l(s(w_t,w_c)) + \sum_{n \in N_{t,c}}l(-s(w_t,n))} $$

在word2vec中对于一个词$w$会有两种词向量表示$u_w,v_w$，其中$u_w$表示输入词向量，$v_w$表示输出词向量（也就是Huffman树进行分类时的参数$\theta$），那么现在我们对于中心词$w_t$和周围词$w_c$有了词向量$u_{w_t}$和$v_{w_c}$，在word2vec中的$s()$函数的计算方法就是：
$$s(w_t,w_c) = u^T_{w_t}v_{w_c}$$

### FastText
前面已经说了Fasttext是基于word2vec进行的改进，它的改进主要是对函数$s(w_t,w_c)$的改进。
在fasttext对一个词统计了它的character n-grams信息，对于一个词$where$，当n=3时，计算它的character级别的3-gram，可以得到：
$$<wh, whe, her, ere,re>$$
对于单一的单词如$her$用$<her>$表示，$<$和$>$表示前缀和后缀。
当然除了它的character n-gram信息外，还要它整个单词的表示$<where>$。
在具体训练阶段，作者统计了n从3到6的所有组合。假设我们现在已经有了一个词的n-gram的集合G.(G中还包括了词自己的完整表示)，用$z_g$来表示每个n-gram $\quad g$，那么$s(w,c)$的计算方法如下：
$$s(w,c) = \sum_{g\in G}z_g^Tv_c$$

采取n-gram的方法，对于那些出现次数比较少的词，也能学到更好的表示。同时为了节省空间，会将n-gram进行hash（共有k个哈希值，k为$2.10^6$），hash到同一位置的多个n-gram会共享一个embedding.
