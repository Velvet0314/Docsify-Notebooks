# 第五章&ensp;生成学习算法

学习至此，我们对于机器学习的学习过程有了一个大致的了解。我们之前所学习到的算法，总是对给定 $ x $ 时的 $ y $ 的条件分布进行建模。例如，逻辑回归中 $ p(y|x; \theta) $ 作为 $ h_\theta(x) = g(\theta^T x) $，其中 $ g $ 是 sigmoid 函数。

试图直接学习 $ p(y|x) $（例如逻辑回归）的算法，或者试图从输入空间 $ \mathcal{X} $ 直接学习映射到标签 $\{0, 1\}$ 的算法（如感知机）被称为 **判别学习算法（discriminative learning algorithms）**。而在刚才我们所举的例子中，是对 $ p(x|y) $（和$ p(y) $）进行建模的算法。这种算法被称为 **生成学习算法（generative learning algorithms）**。

**判别学习算法试图直接学习 $ p(y|x) $，而生成学习算法试图学习 $ p(x|y) $ 和 $ p(y) $。**

通过下面的学习，应该对以下知识有着基本的了解：

* 多元正态分布
* 拉普拉斯平滑

通过下面的学习，应该重点掌握：

* 什么是判别学习算法
* 什么是生成学习算法
* 高斯判别分析 GDA
* 朴素贝叶斯

- - -

### 高斯判别分析

我们首先将要接触的生成学习算法是 **高斯判别分析（Gaussian discriminant analysis）**。

在此之前，我们需要进行一些数学上的准备。

#### 数学准备

一般地，在之前的算法中，我们试图得到 $ p(y|x) $。已知自变量，来预测因变量，这种概率被称为 **后验概率（class posterioris）**。在生成学习算法中，已知因变量，来预测自变量，这种概率被称为 **先验概率（class prioris）**。

> **<font size = 4>先验与后验</font>**<br>
> **简单地讲，先验一般指的是经验性（旧信息），而后验指的是考虑新信息。**

通过贝叶斯公式求得 $ p(y|x) $：

<div class="math">
$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$
</div>

> **<font size = 4>贝叶斯公式</font>**<br>
> **贝叶斯公式是贝叶斯学派的核心公式，它将先验概率和后验概率联系起来。**<br>
> **设试验 $ E $的样本空间为 $ S $， $ A $ 为 $ E $ 的事件，$ B_1,B_2,...,B_n $ 为 $ S $ 的一个划分，且 $ p(A) > 0,p(B_i) > 0 $，有：**<br>
>**$$
>p(B_i|A) = \frac{p(A|B_i)p(B_i)}{p(A)}
>$$**
>**其中 $ p(A) = \sum\limits_{j=1}^{n}p(A|B_j)p(B_j)$，由全概率公式：$ p(A) = p(A|B_1)p(B_1) + ... + p(A|B_n)p(B_n) $ 导出**

由于我们关心的是 $ y $ 离散结果中哪一个的概率更大，而不是要求得具体的概率，所以上面的公式我们可以表达为：

<div class="math">
$$
\arg \max_y p(y|x) = \arg \max_y \frac{p(x|y)p(y)}{p(x)} = \arg \max_y p(x|y)p(y)
$$
</div>

$ p(x) = 1 $ 表示 $ p(x) $ 涵盖了所有的情况。

#### * 多元正态分布

多元正态分布是由 **均值向量（mean vector）** $ \displaystyle{\mu \in \mathbb{R}^n} $ 与一个 **协方差矩阵（covariance matrix）** $ \displaystyle{\Sigma \in \mathbb{R}^{n \times n}} $ 确定的。符号记为 $ \mathcal{N}(\mu,\Sigma) $：

<div class="math">
$$
p(x) = \frac{1}{(2 \pi)^{n/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right)
$$
</div>

其中，$ \Sigma $ 是对称半正定的，$ |\Sigma| $ 表示矩阵 $ \Sigma $ 的行列式。

对于一个随机变量 $ X $，分布在 $ \mathcal{N}(\mu,\Sigma) $ 上，有：

<div class="math">
$$
E[X] = \int_{x}xp(x;\mu,\Sigma)dx=\mu
$$
</div>

一个基于向量的随机变量 $ Z $，其协方差定义为：

<div class="math">
$$
\begin{aligned}
Cov(Z) &= E[(Z-\mu)(Z-\mu)^T] \\[5pt]
&= E[ZZ^T] - E[Z]E[Z]^T
\end{aligned}
$$
</div>

所以，如果 $ X~\mathcal{N}(\mu,\Sigma) $，有：

<div class="math">
$$
Cov(X) = \Sigma
$$
</div>

下面我们给出可视化的图形来帮助我们掌握 $ \Sigma $ 对多元正态分布的影响。

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/06/14/pkd5H74.png" data-lightbox="image-5" data-title="Gaussian distribution 1">
  <img src="https://s21.ax1x.com/2024/06/14/pkd5H74.png" alt="Gaussian distribution 1" style="width:100%;max-width:1000px;cursor:pointer">
 </a>
</div>

接下来我们讨论都是在 $ 2\times1\ 大小的\ 0向量\ \mu \ 与\ 2\times2\ 大小的\ \Sigma $ 的条件下进行的。

左边的图像是 $ \Sigma = I $，我们称为标准正态分布；中间的图像是 $ \Sigma = 0.6I $；右边的图像是 $ \Sigma = 2I $。

由图像我们得知，$ \Sigma $ 的主对角线的大小决定了 $ \mathcal{N}(\mu,\Sigma) $ 是 **密集（compressed）** 还是 **扁平（spread-out）**。

然后，我们试着改变副对角线的值：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/06/14/pkdIQEQ.png" data-lightbox="image-5" data-title="Gaussian distribution 2">
  <img src="https://s21.ax1x.com/2024/06/14/pkdIQEQ.png" alt="Gaussian distribution 2" style="width:100%;max-width:1000px;cursor:pointer">
 </a>
</div>

在均值是 0 的情况下，给出 $ \Sigma $（从左至右）：

<div class="math">
$$
\Sigma = \begin{bmatrix}
 1 & 0 \\
 0 & 1
\end{bmatrix};\ 
\Sigma = \begin{bmatrix}
 1 & 0.5 \\
 0.5 & 1
\end{bmatrix};\ 
\Sigma = \begin{bmatrix}
 1 & 0.8 \\
 0.8 & 1
\end{bmatrix}
$$
</div>

进一步对副对角线的值进行尝试：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/06/14/pkdIrCR.png" data-lightbox="image-5" data-title="Gaussian distribution 3">
  <img src="https://s21.ax1x.com/2024/06/14/pkdIrCR.png" alt="Gaussian distribution 3" style="width:100%;max-width:1000px;cursor:pointer">
 </a>
</div>

$ \Sigma $ 为：

<div class="math">
$$
\Sigma = \begin{bmatrix}
 1 & -0.5 \\
 -0.5 & 1
\end{bmatrix};\ 
\Sigma = \begin{bmatrix}
 1 & -0.8 \\
 -0.8 & 1
\end{bmatrix};\ 
\Sigma = \begin{bmatrix}
 3 & 0.8 \\
 0.8 & 1
\end{bmatrix}
$$
</div>

由此我们得出结论：当对副对角线的值进行变动时，其绝对值越大，朝向 $ 45°/135° $ 分布越密集，且当数值大于 0 时朝向 $ 45° $，小于 0 时朝向 $ 135° $。

最后我们对均值 $ \mu $ 进行变动：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/06/14/pkdI6v6.png" data-lightbox="image-5" data-title="Gaussian distribution 4">
  <img src="https://s21.ax1x.com/2024/06/14/pkdI6v6.png" alt="Gaussian distribution 4" style="width:100%;max-width:1000px;cursor:pointer">
 </a>
</div>

$ \mu $ 为：

<div class="math">
$$
\mu = \begin{bmatrix}
 1 \\
 0
\end{bmatrix};\ 
\mu = \begin{bmatrix}
 -0.5 \\
 0
\end{bmatrix};\ 
\mu = \begin{bmatrix}
 -1 \\
 -1.5
\end{bmatrix};\ 
$$
</div>

显然，$ \mu $ 的改变只是让图像进行了平移。

#### GDA 模型

#### Extra: GDA 与 逻辑回归