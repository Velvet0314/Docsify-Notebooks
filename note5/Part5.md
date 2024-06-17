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

显然，$ \mu $ 的改变只是让分布进行了平移。

#### GDA 模型

如果我们目前遇到了一个分类问题，并且样本的值是连续的，那么我们就可以利用 GDA 模型。假设 $ p(x|y) $ 满足多元正态分布：

<div class="math">
$$
\begin{aligned}
y &\sim \text{Bernoulli}(\phi) \\[5pt]
x|y = 0 &\sim \mathcal{N}(\mu_0, \Sigma) \\[5pt]
x|y = 1 &\sim \mathcal{N}(\mu_1, \Sigma)
\end{aligned}
$$
</div>

其概率分布为：

<div class="math">
$$
\begin{aligned}
p(y) &= \phi^y(1 - \phi)^{1-y} \\[5pt]
p(x|y = 0) &= \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_0)^T\Sigma^{-1}(x - \mu_0)\right) \\[5pt]
p(x|y = 1) &= \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_1)^T\Sigma^{-1}(x - \mu_1)\right)
\end{aligned}
$$
</div>

其中参数为 $ \displaystyle{\phi ,\ \Sigma ,\ \mu_0 ,\ \mu_1 }$。

!> **注意这里的参数有两个 $ \mu $，表示在不同的结果模型下，特征均值不同，但我们假设协方差相同。反映在图上就是不同模型中心位置不同，但形状相同。这样就可以用直线来进行分隔判别。**

给出其对数似然函数：

<div class="math">
$$
\begin{aligned}
\quad \quad \quad \quad\quad \quad \quad \quad \quad \quad \quad \quad \ell(\phi, \mu_0, \mu_1, \Sigma) &= \log \prod_{i=1}^m p(x^{(i)},y^{(i)};\phi, \mu_0, \mu_1, \Sigma) \\[5pt]
&=  \log \prod_{i=1}^m \color{orange}{p(x^{(i)}|y^{(i)}; \mu_0, \mu_1, \Sigma) p(y^{(i)}; \phi)} \\[5pt]
&\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \color{red}{\rightarrow拆分成了一个二维正态分布和一个伯努利分布} \\[5pt]
\end{aligned}
$$
</div>

同样地，为了最大化似然函数，得到参数：

<div class="math">
$$
\begin{aligned}
\phi &= \frac{1}{m} \sum_{i=1}^{m} 1\{y^{(i)} = 1\}\\[5pt]
\mu_0 &= \frac{\sum_{i=1}^{m} 1\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^{m} 1\{y^{(i)} = 0\}}\\[5pt]
\mu_1 &= \frac{\sum_{i=1}^{m} 1\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^{m} 1\{y^{(i)} = 1\}}\\[5pt]
\Sigma &= \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T\\[5pt]
\end{aligned}
$$
</div>

实际上，通过对学习过程的可视化，我们可以看到 GDA 是这样进行学习的：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/06/17/pk0lYgU.png" data-lightbox="image-5" data-title="GDA">
  <img src="https://s21.ax1x.com/2024/06/17/pk0lYgU.png" alt="GDA" style="width:100%;max-width:500px;cursor:pointer">
 </a>
</div>

#### Extra: GDA 与 逻辑回归

如果我们把 $ p(y = 1 | x; \phi, \Sigma, \mu_0, \mu_1) $ 看作是有关 $ x $ 的函数，那么有：

<div class="math">
$$
p(y = 1 | x; \phi, \Sigma, \mu_0, \mu_1) = \displaystyle{\frac{1}{1 + \exp(-\theta^T x)}}
$$
</div>

其中，$ \theta $ 有关是 $ \phi, \Sigma, \mu_1, \mu_0 $ 的函数。这显然是一个逻辑回归的形式。

通常，GDA 与 逻辑回归会给出不同的决策边界。那么我们应该如何正确地选择呢？

首先，我们了解到，GDA 在有 $ p(x|y) $ 是多元正态分布下，其 $ p(y|x) $ 是遵循逻辑回归的假设函数的；但相反地，$ p(y|x) $ 是逻辑函数并不意味着 $ p(x|y) $ 是多元正态分布。这表明了 GDA 的建模相比于逻辑回归是更有力、约束更强的。

进一步地讲，当 $ p(x∣y) $ 确实是高斯分布（具有共享的 $ \Sigma $）时，GDA 是 **渐近有效** 的。也就是说，针对准确估计 $ p(y|x) $，GDA 的效果是最好的。

> **渐进有效性（Asymptotically efficient）**<br>
> **一个估计量是渐进有效的，意味着在样本数量无限增加的情况下，该估计量达到了最小方差，即在所有可能的无偏估计量中，它具有最小的误差。这种估计量在大样本条件下是最精确的。**<br>
> **简单地讲，渐进有效性就是指在数据量非常大的情况下，这种估计方法是最优的，能给出最准确的结果。**

相反，通过做出显著较弱的假设，逻辑回归更为 **鲁棒（robust）**，对不正确的建模假设也更不敏感。有许多不同的假设集会导致 $ p(y|x) $采取逻辑函数的形式。例如，如果 $ x|y = 0 \sim \text{Poisson}(\lambda_0) $，$ x|y = 1 \sim \text{Poisson}(\lambda_1) $，那么 $ p(y|x) $ 将是逻辑函数。逻辑回归在这样的泊松数据上也会表现良好。但如果我们在这种数据上使用 GDA，并将高斯分布拟合到这种非高斯数据上，那么结果将会不太可预测，而 GDA 可能（也可能不会）表现良好。

总的来说，GDA 做出了更强的建模假设，并且在数据效率上更高（即需要更少的训练数据就能“学得好”），当这些建模假设是正确或至少近似正确时。逻辑回归做出了更弱的假设。具体来说，当数据确实是非高斯分布时，在大规模数据集的极限下，逻辑回归几乎总是会比 GDA 表现更好。因此，实际使用中逻辑回归比 GDA 更常用。

### 朴素贝叶斯

朴素贝叶斯是一种基于贝叶斯定理的分类算法。

### 朴素贝叶斯