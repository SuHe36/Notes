1.下面哪一个不是行列式$\begin{vmatrix}
     2 &0  &0 \\
     0 &4  &5 \\
     0 &4  &3
\end{vmatrix}$的特征值（）：
* A. -1
* B. 1
* C. 2
* D. 8

解析：
对上面的行列式进行求特征值：
$\begin{vmatrix}
     2 &0  &0 \\
     0 &4  &5 \\
     0 &4  &3
\end{vmatrix}$ - $\begin{vmatrix}
     \lambda &0  &0 \\
     0 &\lambda  &0 \\
     0 &0  &\lambda
\end{vmatrix}$ = $\begin{vmatrix}
     2-\lambda &0  &0 \\
     0 &4-\lambda  &5 \\
     0 &4  &3-\lambda
\end{vmatrix}$ = $(2-\lambda)[(4-\lambda)(3-\lambda)-5\times 4]=-(\lambda +1)\times(\lambda -2)\times(\lambda -8)$ 
 
所以特征值为$\lambda_1=-1, \lambda_2 = 2,\lambda_3 = 8$。
 
 答案选B



2.从一副52张扑克牌中，随机挑选一张，它是红色的或者是6的概率是多少（）
* A. 15/26
* B. 7/13
* C. 1/2
* D. 6/13

解析： 
 - 选到红色牌的概率：26/52
 - 选到6的概率为：4/52
 - 选到是红色的也是6的概率为：2/52
 - 所以选到红色或者6的概率为：P(红色 or 6 ) = 26/52 + 4/52 - 2/52 = 28/52 = 7/13

答案选B


3.下面说法不正确的是（）：
- A. 无向图的邻接矩阵是对称矩阵
- B. 一颗满二叉树，高度为h，则它的节点总数为$2^h-1$
- C. 在一棵非空二叉树中，叶子节点的总数比度为2的节点总数多1个
- D. 归并排序的时间复杂度与数据初始分布有关。

解析：归并排序算法的时间复杂度与数据初始分布无关，时间复杂度总是$O(nlogn)$

答案选D


4. 下面说法正确的是（）：
- A. SVM是一种生成式模型
- B. 当采用tanh作为激活函数，输出值为-1.5
- C. 与使用L1正则相比，使用L2正则会使更多的参数值为0
- D. softmax(X+c)的结果与softmax(X)的结果一致，其中X是向量，c是常量

解析：SVM是一种判别式模型；tanh的取值范围为(-1,1); L1正则会使更多的参数为0，进行特征选择；
答案选D

1. 下面的描述不正确的是():
- A. 采用CBOW训练词向量，是用周围词去预测中心词
- B. word2vec采用了Hierarchical softmax方法，时间复杂度为$O(N)$
- C. Glove利用词共现矩阵来训练词向量
- D. fasttext引入了词内的n-gram信息

解析：word2vec采用了Hierarchical softmax方法后，时间复杂度由$O(N)$降为$O(logN)$.
答案选B

