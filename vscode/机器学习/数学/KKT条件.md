# KKT条件
KKT条件是非线性规划最佳解的必要条件，**KKT条件将拉格朗日乘数法所处理的等式约束优化问题推广到不等式。**


## 等式约束优化问题
给定一个目标函数$f:R^n \rightarrow R$，我们希望找到$x \in R^n$，在满足约束条件$g(x)=0$的前提下，使得f(x)有最小值，这个约束优化问题记为：
$$
\begin{aligned}
    &\min \qquad f(x) \\
    & s.t. \qquad g(x)=0
\end{aligned}
$$

为了方便分析，假设f和g是连续可导函数，拉格朗日乘数法是等式约束优化问题的典型解法，定义拉格朗日函数：
$$L(x,\lambda) = f(x) + \lambda g(x) $$
其中$\lambda$称为拉格朗日乘数，**拉格朗日乘数法将原本的约束优化问题转换成等价的无约束优化问题。**
$$\mathop{\min}\limits_{x,\lambda} L(x,\lambda)$$
计算L对$x$与$\lambda$的偏导数并设为零，**就可以得到最优解的必要条件：**
$$
\begin{aligned}
    &\nabla_{x}L=\frac{\partial L}{\partial x} = \nabla f + \lambda \nabla_{g} = 0\\
    &\nabla_{\lambda}L=\frac{\partial L}{\partial \lambda} = g(x) = 0 \\
\end{aligned}
$$
**其中第一个为定常方程式(stationary equation)，第二个为约束条件。解开上面的n+1个方程式就可以得到$L(x,\lambda)$的stationary point $x^*$，以及$\lambda$的值（正负皆有可能）**

## 不等式约束优化问题

**接下来我们将约束等式$g(x)=0$推广到不等式$g(x) \leq 0$考虑这个问题：**
$$
\begin{aligned}
    &\min \qquad f(x) \\
    &s.t. \qquad g(x) \leq 0 
\end{aligned}
$$
**约束不等式$g(x) \leq 0$称为原始可行性，据此，我们定义可行域$K=x \in R^n|g(x) \leq 0$.**
**我们先给出其主要思想：将不等书约束条件变成等式约束条件，具体做法就是引入松弛变量，其中松弛变量也是优化变量，也需要一视同仁对齐进行求偏导。**


**对于约束g(x)我们可以引入一个松弛变量$a^2$，得到$g(x)+a^2=0$，注意这里直接加上平方项$a^2$，而不是a，是因为g(x)这个不等式的左边必须加上一个正数才能使不等式变为等式，若只加上a，又会引入新的约束$a>0$，这不符合我们的意愿，所以这样的话，我们就可以将不等式约束转换为等式约束：**
$$
\begin{aligned}
    &\min \qquad f(x) \\
    &s.t. \qquad g(x) +a^2 =  0 
\end{aligned}
$$
**有了这个等式约束，我们就可以得到拉格朗日函数了：**
$$L(x,a,\lambda) = f(x) + \lambda(g(x) + a^2) $$
然后我们再按照等式约束优化问题（极值必要条件）对其求解，联立方程：
$$
\begin{cases}
    \frac{\partial L}{\partial x} = \frac{\partial f}{\partial x} + \lambda \frac{\partial g}{\partial x} = 0 \\
    \frac{\partial L}{\partial \lambda} = g(x) + a^2 =0 \\
    \frac{\partial L}{\partial a} = 2\lambda a = 0 \\
    \lambda \geq 0
\end{cases}
$$
**这里默认$\lambda \geq 0$，这里不进行解释，具体的解释涉及到用几何性质【其实是作者的空间想象力匮乏，看不懂，嗯】，其实对于所有的不等式约束前的乘子，我们都要求其大于等于0.**
得到方程组后，我们就开始动手解它，看到第三行的式子$2\lambda a =0$比较简单，就可以直接从它入手。
对于$2\lambda a =0$有两种情况：
1. $\lambda = 0, a \neq 0$：此时，由于乘子$\lambda=0$，因此g(x)与其相乘，得到的结果均为0，可以理解为约束不起作用，且有$g(x)=-a^2<0$
2. $a=0, \lambda \geq 0$:此时，由于$g(x) + a^2 = 0，且a=0$，所以g(x)=0，这时可以理解为约束g(x)在起作用。

**综上两种情况，可以得出结论：$\lambda a =0$，在约束起作用时$g(x)=0,\lambda \geq 0$，在约束不起作用时$g(x)<0, \lambda=0$。**

**因此，上面的方程组（极值必要条件），可以转换为：**
$$
\begin{cases}
    \frac{\partial L}{\partial x} = \frac{\partial f}{\partial x} + \lambda \frac{\partial g}{\partial x} = 0 \\
    \lambda g(x) = 0 \\
    \lambda \geq 0 
\end{cases}
$$
这是一元一次的形式，类似的，对于多元不等式约束问题:
$$
\begin{aligned}
    &\min \quad f(x) \\
    &s.t. \quad g_j(x) \leq 0, \quad j=1,2,...,m
\end{aligned}
$$
我们有：
$$
\begin{cases}
    \frac{\partial f}{\partial x_i} + \sum_{j=1}^{m}\lambda_j\frac{\partial g_j(x)}{\partial x_i} = 0, \quad i=1,2...n \\
    \lambda_jg_j(x) = 0 \quad j=1,2,...,m \\
    \lambda_j \geq 0 \quad j=1,2,...m
\end{cases}
$$

**上式就是不等式约束优化问题的KKT条件，$\lambda_j$称为KKT乘子，且约束起作用时$g_j(x)=0,\lambda_j \geq 0$;约束不起作用时$g_j(x) < 0, \lambda_j =0$**


**因此，可将上式推广到多个约束等式与约束不等式的情况，考虑如下的标准约束优化问题，也叫做非线性规划：**
$$
\begin{aligned}
    &\min \quad f(x) \\
    &s.t.\quad g_j(x) \leq 0, \quad j=1,2,...,m \\
    & \qquad h_k(x) = 0, \quad k=1,2,...,p \\
\end{aligned}
$$
定义拉格朗日函数：
$$ L(x,\lambda_j, \mu_k) = f(x) + \sum_{j=1}^{m}\lambda_jg_j(x) + \sum_{k=1}^p\mu_kh_k(x)$$
其中，$\lambda_j$是对应$g_j(x) \leq 0$的拉格朗日乘数，也叫做KKT乘数，$\mu_k$是对应$h_k(x)=0$的拉格朗日乘数，KKT条件包括：
$$
\begin{aligned}
    &\nabla_xL = 0 \\
    &h_k(x) =0, \quad k=1,2,...,p \\
    &g_j(x) \leq 0, \quad j=1,2,...,m\\
    &\lambda_j \geq 0,\quad j=1,2,...,m \\
    &\lambda_jg_j(x)=0,\quad j=1,2,...,m
\end{aligned}
$$
**因此KKT条件中包括**：
- 拉格朗日函数的定长方程式：$\nabla_xL = 0 $
- 原始可行性：$h_k(x) =0,g_j(x) \leq 0$
- 对偶可行性：$\lambda_j \geq 0$
- 互补松弛性：$\lambda_jg_j(x)=0$

参考网址1：https://zhuanlan.zhihu.com/p/38163970
参考网址2：https://zhuanlan.zhihu.com/p/26514613