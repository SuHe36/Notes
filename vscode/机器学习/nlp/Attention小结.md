Attention，一般都是在encoder-decoder框架中进行使用，
我们常使用的方法就是decoder的隐状态$d_t$和encoder的每一个隐状态算一个分值，然后再用softmax归一化。

但这个计算分值的过程，我们常使用的就是`dot`运算，也就是对位相乘，其实还有其他的几种方法，
这些方法有一个统一的名称：对齐方式。

$$score(h_e,h_d) =\begin{cases}
    &h_e^Th_d \quad    & dot \\
    &h_e^TW_ah_d \quad & general \\
    &v_a^Ttanh(W_a[h_e;h_d]) &concat
\end{cases}$$

其中$score(h_e,h_d)$表示源端和目标单词的对齐程度，所以常用的对齐方式就是上面的三种：`dot product、general、concat`

## soft attention 和hard attention
- soft attention：就是采取前面的计算score的方法，对每一个encoder的隐向量算一个分值，最后把这些分值和对应的encoder的隐向量相乘再相加，得到一个糅合attention信息的表示。**所以soft attention是参数化的，是可导的，可以直接嵌入到模型中去，直接训练。梯度是可以经过Attention Mechanism模块，然后反向传播到模型中的其他部分去的。**

- hard attention：hard attention是一个随机的过程。前面计算score的过程，soft attention和hard attention没有特别大的出入。**在计算出score后，hard attention不会选择整个encoder的输出作为输入，而是把score视作概率值来采样输入端的某个隐状态来进行计算。为了实现梯度的反向传播，需要采用蒙特卡洛采样的方法来估计模块的梯度。【强化学习这块不太懂，后续可以再看】**

两种attention都有各自的优势，但是目前更多的研究和应用都是更加倾向于使用soft attention，因为其可以直接求导，进行梯度反向传播。

## global attention 和 local attention
- global attention：和传统的attention model一样，encoder的所有的hidden state都被用于计算context vector的权重，然后计算出score分布后，用softmax归一化，最后再用加权平均的方式，得到最终的揉入attention信息的表示（也叫做上下文向量$c_t$）。


- local attention：global attention有一个明显的缺点，就是每一次encoder端的所有hidden state 都要参与运算，这样做计算开销会比较大，特别是当encoder的句子偏长，比如一句话或者一篇文章，这样的效率就会很低。因此，为了提高效率，local attention就应运而生。
  - **local attention首先会为decoder端当前的词，预测一个source端的对齐位置$p_t$，然后基于$p_t$选择一个窗口，用于计算最终的上下文向量$c_t$。位置$p_t$的计算方式如下：**
  $$p_t = S \cdot sigmoid(v_p^T tanh(W_ph_t))$$
  - 其中$S$是encoder端的句子长度，$v_p$和$w_p$是模型参数。此时，对齐向量$a_t$（也就是score的数值）的计算公式如下：
  $$a_t(s) = align(h_e,h_d)exp(-\frac{(s-p_t)^2}{2\sigma^2})$$
  - 计算出每个向量的分值$a_t$后，然后最终的上线文向量$c_t$，选取的encoder端的隐藏状态只是在一个窗口内$[p_t-D, p_t+D]$，其中的$p_t$是前面计算出来的，$D$是按照经验值选取的。


global attention和local attention各有优劣，实际中global用的更多一点，因为：
- local attention当encoder不长时，计算量并没有减少；
- 位置向量$p_t$的预测准不准确，直接影响到local attention的准确率；

## self attention
self attention与传统的attention机制并不相同，传统的attenton是基于source端和target端的隐变量计算出attention的，得到的结果是源端的每个词与目标端的每个词之间的依赖关系。

**但是self attention不同，他是在source端和target端分别独立进行的，仅与source input或者target input自身相关的self attention，捕捉source端和target端自身的词与词之间的依赖关系；然后再把source端得到的self attention信息加入到target端得到的attention中，捕捉source端和target端词与词之间的依赖关系。**

因此self attention比传统的attention mechanism的效果更好，主要原因就是，传统的attention机制忽略了源端或目标端句子中词与词之间的依赖关系,相对比，self attention不仅可以得到源端和目标端词与词之间的依赖关系，同时还可以有效获取源端和目标端自身词与词之间的依赖关系。

self attention的著名应用就是在transformer里。transformer里的self attention是在scaled dot-product attention单元里面实现的，首先把输入input经过线性变换后分别得到Q、K、V，**注意，Q,K,V都来自于input，只不过是线性变换的矩阵的权值不同而已。**然后把Q和K做dot product相乘，得到输入input词与词之间的依赖关系，然后在经过尺度变换(scale)、掩码(mask)和softmax操作，得到最终的self attention矩阵。

**尺度变换是为了防止输入值过大导致训练不稳定，mask则是为了保证时间的先后关系。**

最后，把encoder端self attention计算的结果加入到decoder中作为k个v，结合decoder自身的输出作为q，得到encoder端的attention与decoder端attention之间的依赖关系。


## attention的其他一些组合使用
### Hierarchical Attention

Hierarchical attention提出构建两个层次的attention mechanism，第一个层次是对句子中每个词的attention，第二个层次是针对文档中的每个句子的的attention。

主要实现方法就是，用GRU对每个词获取一个隐层表示，然后通过一个线性层变换一下，还随机初始化一个向量$u_w$，然后可以使用`dot product`计算每个词的隐层表示与向量$u_w$的一个相似度，最后在用softmax归一化这个相似度，然后就可以用这个相似度和对应词的隐层表示相乘再相加得到一个句子级别的表示。

得到句子级别的表示后，同样可以经过一个GRU，得到新的句子级别的表示，然后用上面同样的方法，计算一个对各个句子的一个attention分布，然后用这个attention分布与对应的句子向量相乘再相加，得到一个文档表示。

### Synthesizer attention
SYNTHESIZER: Rethinking Self-Attention in Transformer Models

Transformer中采取的attention方式是quer-key-value的dot-product的方式，其核心就是单个token相对于序列中的其他token的重要性。

本文对于这一做法提出了质疑，本文认为并不需要计算token-token之间的重要性，作者提出了一个Synthetic Attention。

主要有两种:

- dense synthesizer：生成的attention矩阵是由输入x经过两个dense层映射而来
- random synthesizer：生成的attention矩阵是随机初始化得到的，然后在训练的过程中进行优化；
作者还提出了将transformer attention和两种synthesizer attention糅合的方法，实验结果显示三种attention在机器翻译任务上相差不多，

但是在对话生成任务上，synthesizer attention的效果要比transformer attention的效果好，而在finetune模式上，则是采取transformer attention的效果好。

作者主要是提出了对于transformer attention这种采取token-token的方式计算attention的质疑，并用实验验证了结果，

但是为什么这几种attention的效果有差异或者相似，作者并未给出说明。主要是对当前主流attention机制提出了探索和思考。

![](../../figure/100.png)
