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

> [!NOTE]
> **<font size = 4>先验与后验</font>**<br>
> **简单地讲，先验一般指的是经验性（旧信息），而后验指的是考虑新信息。**

通过贝叶斯公式求得 $ p(y|x) $：

<div class="math">
$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$
</div>

> [!NOTE]
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

所以，如果 $ X\sim\mathcal{N}(\mu,\Sigma) $，有：

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

> [!WARNING]
> **注意这里的参数有两个 $ \mu $，表示在不同的结果模型下，特征均值不同，但我们假设协方差相同。反映在图上就是不同模型中心位置不同，但形状相同。这样就可以用直线来进行分隔判别。**

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

> [!TIP]
> **<font size=4>渐进有效性（Asymptotically efficient）</font>**<br>
> **一个估计量是渐进有效的，意味着在样本数量无限增加的情况下，该估计量达到了最小方差，即在所有可能的无偏估计量中，它具有最小的误差。这种估计量在大样本条件下是最精确的。**<br>
> **简单地讲，渐进有效性就是指在数据量非常大的情况下，这种估计方法是最优的，能给出最准确的结果。**

相反，通过做出显著较弱的假设，逻辑回归更为 **鲁棒（robust）**，对不正确的建模假设也更不敏感。有许多不同的假设集会导致 $ p(y|x) $采取逻辑函数的形式。例如，如果 $ x|y = 0 \sim \text{Poisson}(\lambda_0) $，$ x|y = 1 \sim \text{Poisson}(\lambda_1) $，那么 $ p(y|x) $ 将是逻辑函数。逻辑回归在这样的泊松数据上也会表现良好。但如果我们在这种数据上使用 GDA，并将高斯分布拟合到这种非高斯数据上，那么结果将会不太可预测，而 GDA 可能（也可能不会）表现良好。

总的来说，GDA 做出了更强的建模假设，并且在数据效率上更高（即需要更少的训练数据就能“学得好”），当这些建模假设是正确或至少近似正确时。逻辑回归做出了更弱的假设。具体来说，当数据确实是非高斯分布时，在大规模数据集的极限下，逻辑回归几乎总是会比 GDA 表现更好。因此，实际使用中逻辑回归比 GDA 更常用。

### 朴素贝叶斯

在 GDA 中，特征向量 $ x $ 是连续的实数向量。现在我们将讨论 $ x $ 是离散的情况，这将会会运用到另一种学习算法。

**朴素贝叶斯（naive bayes）** 是一种基于贝叶斯定理的分类算法。下面，我们将通过垃圾邮件分类的例子来进一步学习朴素贝叶斯。

将一封邮件作为输入特征向量，与现有的字典进行比较，如果在字典中第 $ i $ 个词在邮件中出现，$ x_i = 1 $，否则 $ x_i = 0 $。假设输入特征向量将表示为：

<div class="math">
$$
x = \left[\begin{array}{@{}c@{}}
\ 1 \  \\
\ 0 \  \\
\ 0 \  \\
\ \vdots \  \\
\ 1 \  \\
\ \vdots \  \\
\ 0 \  \\
\end{array}\right] \hspace{10mm}
\begin{array}{l}
\text{a} \\
\text{aardvark} \\
\text{aardwolf} \\
\vdots \\
\text{buy} \\
\vdots \\
\text{zygmurgy} \\
\end{array}
$$
</div>

现在，对 $ p(x|y) $ 进行建模。

#### NB 假设

假设字典中有 50000 个词，$ x \in \\{0, 1\\}^{50000} $ 如果采用多项式建模，将会有 $ 2^{50000} $ 种结果，$ 2^{50000} - 1 $ 维的参数向量，这样参数会明显过多。所以为了对$ p(x|y) $ 建模，需要做一个强假设，假设 $ x $ 的特征是条件独立的，这个假设称为 **朴素贝叶斯假设（naive bayes (NB) assumption）** ,导出的算法称为 **朴素贝叶斯分类（naive bayes classifier）**。

> [!TIP]
> **朴素贝叶斯分类器被称为“朴素”是因为它假设特定条件下各特征之间相互独立。<br> 例如，如果 $ y = 1 $ 代表垃圾邮件，"buy"是第 2087 个词，"price"是第 39831 个词，假设某封邮件是垃圾邮件（即 $ y = 1 $），知道邮件中出现了"buy"这个词，不会影响"price"这个词出现与否。进一步地，这可以写成：**
**$$ p(x_{2087} \mid y) = p(x_{2087} \mid y, x_{39831}) $$**
**注意，这不是说 $ x_{2087} $ 和 $ x_{39831} $ 是独立的，否则会写为 $ p(x_{2087}) = p(x_{2087} \mid x_{39831}) $。<br> 实际上，我们只是假设在给定 $ y $ 的条件下 $ x_{2087} $ 和 $ x_{39831} $ 是条件独立的。**

通过 NB 假设，我们可以得到:

<div class="math">
$$
\begin{aligned}
p(x_1, \ldots, x_{50000} \mid y) &= p(x_1 \mid y)p(x_2 \mid y, x_1)p(x_3 \mid y, x_1, x_2) \cdots p(x_{50000} \mid y, x_1, \ldots, x_{49999}) \\[3pt]
&\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \color{red}{\rightarrow概率的基本性质} \\[5pt]
&= p(x_1 \mid y)p(x_2 \mid y)p(x_3 \mid y) \cdots p(x_{50000} \mid y) \\[3pt]
&\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \color{red}{\rightarrow通过\ NB\ 假设等价转换} \\[5pt]
&= \prod_{i=1}^{n} p(x_i \mid y)
\end{aligned}
$$
</div>

得到的模型参数：

<div class="math">
$$
\begin{aligned}
\phi_{i \mid y=1} &= p(x_i = 1 \mid y = 1) \\[5pt]
\phi_{i \mid y=0} &= p(x_i = 1 \mid y = 0) \\[5pt]
\phi_y &= p(y = 1)
\end{aligned}
$$
</div>

对于一个训练集 $ {(x^{(i)}, y^{(i)}) ; i = 1, \ldots, m\} $，写出其 **联合似然函数（joint likelihood）**：

<div class="math">
$$
\mathcal{L}(\phi_{y}, \phi_{j \mid y=0}, \phi_{j \mid y=1}) = \prod_{i=1}^{m} p(x^{(i)}, y^{(i)})
$$
</div>

还是同样地对参数进行极大似然估计：

<div class="math">
$$
\begin{aligned}
\phi_{j \mid y=1} &= \frac{\sum_{i=1}^{m} 1\{x_j^{(i)} = 1 \land y^{(i)} = 1\}}{\sum_{i=1}^{m} 1\{y^{(i)} = 1\}} \ \ \color{red}{\rightarrow在垃圾邮件中，单词\ x_j\ 出现的概率} \\[5pt]
\phi_{j \mid y=0} &= \frac{\sum_{i=1}^{m} 1\{x_j^{(i)} = 1 \land y^{(i)} = 0\}}{\sum_{i=1}^{m} 1\{y^{(i)} = 0\}} \ \ \color{red}{\rightarrow垃圾邮件出现的概率} \\[5pt]
\phi_{y} &= \frac{\sum_{i=1}^{m} 1\{y^{(i)} = 1\}}{m}
\end{aligned}
$$
</div>

其中，符号 $ \wedge $ 表示 "与"。然后，写出概率的公式：
<div class="math">
$$
\begin{aligned}
p(y = 1 \mid x) &= \frac{p(x \mid y = 1)p(y = 1)}{p(x)} \\[5pt]
&= \frac{(\prod_{i=1}^{n} p(x_i \mid y = 1))p(y = 1)}{(\prod_{i=1}^{n} p(x_i \mid y = 1))p(y = 1) + (\prod_{i=1}^{n} p(x_i \mid y = 0))p(y = 0)} \\[3pt]
&\qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \color{red}{\rightarrow全概率公式展开}
\end{aligned}
$$
</div>

#### 特征向量离散化

最后，我们对朴素贝叶斯的应用进行一个推广。

当连续的属性值不太满足多元正态分布时，我们可以通过对其特征向量离散化，利用 NB 来代替 GDA：

<div style="display: table; margin: auto;">
<table style="width: 100%;">
  <tr>
    <th style="text-align: center; border: 1px solid black; padding: 8px;">$\text{Living area (sq. feet)}$</th>
    <th style="text-align: center; border: 1px solid black; padding: 8px;">&lt; $400$</th>
    <th style="text-align: center; border: 1px solid black; padding: 8px;">$400-800$</th>
    <th style="text-align: center; border: 1px solid black; padding: 8px;">$800-1200$</th>
    <th style="text-align: center; border: 1px solid black; padding: 8px;">$1200-1600$</th>
    <th style="text-align: center; border: 1px solid black; padding: 8px;">&gt; $1600$</th>
  </tr>
  <tr>
    <td style="text-align: center; border: 1px solid black; padding: 8px;">$x_i$</td>
    <td style="text-align: center; border: 1px solid black; padding: 8px;">$1$</td>
    <td style="text-align: center; border: 1px solid black; padding: 8px;">$2$</td>
    <td style="text-align: center; border: 1px solid black; padding: 8px;">$3$</td>
    <td style="text-align: center; border: 1px solid black; padding: 8px;">$4$</td>
    <td style="text-align: center; border: 1px solid black; padding: 8px;">$5$</td>
  </tr>
</table>
</div>

#### 拉普拉斯平滑

朴素贝叶斯在大多数情况下都表现良好，但是也存在例外：对稀疏数据问题较为敏感。

比如在邮件分类时，NIPS 这个单词显得太过于高大上，邮件中可能没有出现过，现在新来了一个邮件 "NIPS call for papers"，假设NIPS 这个词在词典中的位置为 35000，然而 NIPS 这个词从来没有在训练数据中出现过，这是第一次出现 NIPS，于是其概率为：

<div class="math">
$$
\phi_{35000|y=1} = \frac{\sum_{i=1}^{m} 1\{x^{(i)}_{35000} = 1 \wedge y^{(i)} = 1\}}{\sum_{i=1}^{m} 1\{y^{(i)} = 1\}} = 0
$$
</div>

<div class="math">
$$
\phi_{35000|y=0} = \frac{\sum_{i=1}^{m} 1\{x^{(i)}_{35000} = 1 \wedge y^{(i)} = 0\}}{\sum_{i=1}^{m} 1\{y^{(i)} = 0\}} = 0
$$
</div>

这是由于 NIPS 从未在垃圾邮件和正常邮件中出现过，所以概率都为 0。那么，最后的后验概率即为：

<div class="math">
$$
\begin{aligned}
\qquad \qquad \qquad \qquad \qquad p(y = 1|x) &= \frac{\prod_{i=1}^{n} p(x_i|y = 1)p(y = 1)}{\prod_{i=1}^{n} p(x_i|y = 1)p(y) + \prod_{i=1}^{n} p(x_i|y=0)p(y = 0)} \\[5pt]
&\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \color{red}{\rightarrow \prod_{i=1}^{n}p(x_i \mid y = 1)中包含了 p(x_{35000} \mid y) = 0} \\[2pt]
&= \frac{0}{0}
\end{aligned}
$$
</div>

也就是出现了"零概率问题"。

> [!NOTE]
> **<font size=4>零概率问题</font>**<br>
> **在进行概率模型计算时，因为训练数据中某些事件未曾出现，导致这些事件的概率被错误地估计为零。由于朴素贝叶斯模型采用概率乘积来计算类别概率，任何一个概率为0的特征都会使得整个类别的概率计算结果为 0，这显然会影响到分类的正确性。**

为了解决零概率问题，通常采用 **拉普拉斯平滑（laplace smoothing）** 解决。

考虑估计一个取值为 $\\{1, ..., k\\}$ 的多项随机变量 $z$ 的均值的问题。我们可以用 $\phi = p(z = i)$ 参数化我们的多项式。给定 $m$ 个独立观测值 $\\{z^{(1)}, ..., z^{(m)}\\}$，最大似然估计给出如下：

<div class="math">
$$
\phi_j = \frac{\sum_{i=1}^m 1\{z^{(i)} = j\}}{m}
$$
</div>

正如我们之前看到的，如果我们使用这些最大似然估计，那么一些 $\phi_j$ 可能最终为零，这是个问题。为了避免这个问题，我们可以使用拉普拉斯平滑，它用以下估计替换上面的估计：

<div class="math">
$$
\phi_j = \frac{\sum_{i=1}^m 1\{z^{(i)} = j\} + 1}{m + k}
$$
</div>

这里，我们在分子中加1，在分母中加k。

> [!WARNING]
> **$\sum_{j=1}^k \phi_j = 1$ 仍然成立，这是一个理想的属性，因为 $\phi_j$ 是我们已知必须求和为 1 的概率估计。此外，$\phi_j \neq 0$ 对所有 $j$ 的值成立，解决了我们的概率估计为零的问题。在某些（可以说是相当强的）条件下，可以证明拉普拉斯平滑实际上提供了最优的估计器。**

回到我们的朴素贝叶斯分类器，并使用拉普拉斯平滑，我们因此获得以下参数估计：

<div class="math">
$$
\phi_{j|y=1} = \frac{\sum_{i=1}^{m} 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 1\} + 1}{\sum_{i=1}^{m} 1\{y^{(i)} = 1\} + 2}
$$
</div>

<div class="math">
$$
\phi_{j|y=0} = \frac{\sum_{i=1}^{m} 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 0\} + 1}{\sum_{i=1}^{m} 1\{y^{(i)} = 0\} + 2}
$$
</div>

> [!TIP]
> **$ k $ 通常取分类种类数**

实际上，是否对 $ \phi_y $ 应用拉普拉斯平滑通常不太重要，因为我们通常会有相当一部分的垃圾邮件和非垃圾邮件，所以 $ \phi_y $ 将是 $ p(y=1) $ 的合理估计，并且会远离零。

#### * 文本分类的事件模型

在结束生成学习算法的讨论之际，让我们谈谈一个专门用于文本分类的模型。虽然朴素贝叶斯模型适用于许多分类问题，但对于文本分类，有一个相关的模型可能表现更佳。

在文本分类的具体环境中，通常使用的朴素贝叶斯模型采用了所谓的多元伯努利事件模型。在这个模型中，我们假设一封邮件的生成方式是首先随机决定由垃圾邮件发送者还是非垃圾邮件发送者发送你的下一条消息。然后，发送邮件的人通过词典运行邮件，决定是否包括每个单词 $ i $，并根据概率 $p(x_i = 1|y)$ 独立决定。因此，一条消息的概率由 $p(y) \prod_{i=1}^n p(x_i|y)$ 给出。这里用的模型称为多元伯努利模型。

为了描述这个模型，我们将使用不同的符号和特征集来表示邮件。现在我们让 $x_i$ 表示邮件中的第 $i$ 个标识符，它取 $\\{1, ..., |V|\\}$ 中的值，其中 $|V|$ 是我们词汇表（字典）的大小。一封包含 n 个词的邮件现在由一个向量 $(x_1, x_2, ..., x_n)$ 表示，长度为 n；注意这可以表示不同的单词。例如，如果一封邮件以 “A NIPS ...” 开始，则 $x_1 = 1$（"a"是词典中的第 1 个词），$x_2 = 35000$（如果 "nips" 是词典中的第 35000 个词）。

在多项式事件模型中，我们假设一封邮件是通过一个随机过程生成的，在这个过程中，首先决定邮件是垃圾邮件还是非垃圾邮件，然后通过从单词的多项式分布中生成 $x_1$ 开始生成邮件，接着独立地选择 $x_2$，但仍来自同一个多项式分布，接下来依此类推直到所有单词都生成。因此，一条消息的总概率由 $p(y) \prod_{i=1}^n p(x_i|y)$ 给出。注意这个公式看起来类似于我们之前对于多元伯努利消息概率的公式，但是公式中的项现在表示的是完全不同的事物。特别是，$x_i|y$ 现在是一个多项式的，而不是伯努利分布的。

我们新模型的参数为 $\phi_y = p(y)$，如前所述，$\phi_{k|y=1} = p(x_j = k | y = 1)$ 对任何 j 和 $\phi_{k|y=0} = p(x_j = k | y = 0)$。注意我们假设 $p(x_j|y)$ 对所有的 j 都是相同的（即单词的生成分布不依赖于它在邮件中的位置）。

如果我们有一个训练集 $\\{(x^{(i)}, y^{(i)}); i = 1, ..., m\\}$，其中 $x^{(i)} = (x_1^{(i)}, x_2^{(i)}, ..., x_{n_i}^{(i)})$（这里，$n_i$ 是第 $i$ 个训练样本中的单词数），则数据的似然度由下式给出：

<div class="math">
$$
\begin{aligned}
L(\phi_{k|y=0}, \phi_{k|y=1}) &= \prod_{i=1}^m P(x^{(i)}, y^{(i)}) \\[5pt]
&= \prod_{i=1}^m \left( \prod_{j=1}^{n_i} P(x_j^{(i)} | y^{(i)}; \phi_{k|y=0}, \phi_{k|y=1}) \right) P(y^{(i)}; \phi_y)
\end{aligned}
$$
</div>

进一步地，得到参数的最大似然估计：

<div class="math">
$$
\begin{aligned}
\phi_{k|y=1} &= \frac{\sum_{i=1}^m \sum_{j=1}^{n_i} 1\{x_j^{(i)} = k \wedge y^{(i)} = 1\}}{\sum_{i=1}^m 1\{y^{(i)} = 1\} n_i} \\[5pt]
\phi_{k|y=0}& = \frac{\sum_{i=1}^m \sum_{j=1}^{n_i} 1\{x_j^{(i)} = k \wedge y^{(i)} = 0\}}{\sum_{i=1}^m 1\{y^{(i)} = 0\} n_i} \\[5pt]
\phi_y &= \frac{\sum_{i=1}^m 1\{y^{(i)} = 1\}}{m}
\end{aligned}
$$
</div>

如果我们应用拉普拉斯平滑（实际中需要以获得良好性能），当估计 $\phi_{k|y=0}$ 和 $\phi_{k|y=1}$ 时，我们在分子中加 1，并在分母中加上 $|V|$（词汇表大小）：

<div class="math">
$$
\begin{aligned}
\phi_{k|y=1} &= \frac{\sum_{i=1}^m \sum_{j=1}^{n_i} 1\{x_j^{(i)} = k \wedge y^{(i)} = 1\} + 1}{\sum_{i=1}^m 1\{y^{(i)} = 1\} n_i + |V|}\\[5pt]
\phi_{k|y=0} &= \frac{\sum_{i=1}^m \sum_{j=1}^{n_i} 1\{x_j^{(i)} = k \wedge y^{(i)} = 0\} + 1}{\sum_{i=1}^m 1\{y^{(i)} = 0\} n_i + |V|}\\[5pt]
\phi_y &= \frac{\sum_{i=1}^m 1\{y^{(i)} = 1\}}{m}
\end{aligned}
$$
</div>

虽然朴素贝叶斯分类器可能不是最佳的分类算法，但它通常表现出奇妙的好效果。它通常也是尝试的 “首选” 算法，因为它简单且易于实现。
