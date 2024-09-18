# 第十章&ensp;因子分析

当我们有数据 <span class="math">$ x^{(i)} \in \mathbb{R}^n $</span> ，来自于若干个混合高斯分布时，EM 算法可以用于拟合高斯混合模型。在这种情况下，我们通常设想问题是数据足够大，以至于能够识别出数据中的多重高斯结构。例如，当训练集大小 <span class="math">$ m $</span> 明显大于数据的维度 <span class="math">$ n $</span> 时。

现在，考虑当 <span class="math">$ m \gg n $</span> 的情形。在这种情况下，用单一的高斯分布拟合数据就变得比较困难，更不用说使用多个高斯分布的混合了。具体而言，由于数据点基本位于 <span class="math">$ \mathbb{R}^n $</span> 的一个低维子空间中，如果我们将数据建模为高斯分布，并使用常见的最大似然估计方法来估计均值和协方差矩阵：

<div class="math">
$$
\begin{aligned}
\mu &= \frac{1}{m} \sum_{i=1}^{m} x^{(i)} \\[5pt]
\Sigma &= \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)(x^{(i)} - \mu)^T
\end{aligned}
$$
</div>

我们会发现矩阵 <span class="math">$ \Sigma $</span> 是奇异的。这意味着其逆矩阵 <span class="math">$ \Sigma^{-1} $</span> 不存在，并且 <span class="math">$ 1/|\Sigma|^{1/2} = 1/0 $</span>。然而，这两者在计算多元高斯分布的似然时是必须的。另一种描述此问题的方式是，对参数的最大似然估计会产生一个高斯分布，其概率分布在由样本数据所张成的仿射空间中，对应着一个奇异的协方差矩阵。

> [!NOTE]
> 这里的样本数据是一个点集。对于某些 <span class="math">$ \alpha_i $</span>，集合中的 <span class="math">$ x $</span> 都满足 <span class="math">$ x = \Sigma^{m}_{i=1} \alpha_i x^{(i)} $</span>，因此有 <span class="math">$ x = \Sigma^{m}_{i=1} \alpha_1 = 1 $</span>.

一般来说，除非 <span class="math">$ m $</span> 超过 <span class="math">$ n $</span> 一定的量，否则通过最大似然估计得到的均值和协方差可能会相当差。然而，我们仍然希望能够通过数据拟合一个合理的高斯模型，并可能捕捉数据中某些有意义的协方差结构。我们该怎么做呢？

在这一章节中，我们会首先回顾对  <span class="math">$ \Sigma $</span>  的两种可能的限制，这些限制允许我们在少量数据下拟合  <span class="math">$ \Sigma $</span> ，但都不能给出令人满意的解决方案。然后，我们将讨论高斯分布的一些特性，这些特性将在后面需要到；特别是，如何找到高斯分布的边缘和条件分布。最后，我们给出 **因子分析模型（factor analysis model）** 及其对应的 EM 算法。

通过下面的学习，应该重点掌握：

* 交叉验证
* 正则化

- - -

### <span class="math">$\Large{\Sigma}$</span> 的约束条件

如果我们没有足够的数据来拟合一个完整的协方差矩阵，我们可以对将要考虑的矩阵  <span class="math">$\Sigma$</span>  的空间施加一些限制。例如，我们可以选择拟合一个对角线上的协方差矩阵  <span class="math">$\Sigma$</span> 。在这种情况下，对角线矩阵  <span class="math">$\Sigma$</span>  的最大似然估计由以下公式给出：

<div class="math">
$$
\Sigma_{jj} = \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2
$$
</div>

因此，<span class="math">$\Sigma_{jj}$</span> 就是数据第 <span class="math">$j$</span> 个坐标的方差的经验估计值。

回顾一下，高斯模型的密度分布的形状是椭圆形的。一个对角线的 <span class="math">$\Sigma$</span> 对应于和该椭圆长轴对齐的高斯分布。

有时，我们可能会对协方差矩阵施加进一步的限制：不仅要求其为对角矩阵，而且要求其对角线元素相等。在这种情况下，我们有 <span class="math">$\Sigma = \sigma^2 I$</span>，其中 <span class="math">$\sigma^2$</span> 是我们控制的参数。对 <span class="math">$\sigma^2$</span> 的最大似然估计可以通过以下公式得到：

<div class="math">
$$
\sigma^2 = \frac{1}{mn} \sum_{j=1}^{n} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2
$$
</div>

该模型对应于密度分布是圆形（在二维中）的高斯分布（在更高维度中是球或超球体）。

如果我们为数据拟合一个完整的、无约束的协方差矩阵 <span class="math">$\Sigma$</span>，则要求 <span class="math">$m \geq n + 1$</span>，以确保 <span class="math">$\Sigma$</span> 的最大似然估计不为奇异矩阵。在上述两种限制条件下，只需 <span class="math">$m \geq 2$</span> 时，就能得到非奇异的协方差矩阵 <span class="math">$\Sigma$</span>。

然而，将 <span class="math">$\Sigma$</span> 限制为对角矩阵也意味着将数据的不同坐标 <span class="math">$x_i, x_j$</span> 模型化为不相关且独立的。然而，我们通常希望能够捕捉到数据中一些有意义的相关结构。如果我们使用上述对 <span class="math">$\Sigma$</span> 的任何限制，我们可能会无法捕捉到这些信息。但在接下来的讲解中，我们将描述因子分析模型，该模型使用的参数比对角 <span class="math">$\Sigma$</span> 更多，并捕捉到数据中的一些相关性信息。尽管如此，也无法完全拟合完整的协方差矩阵。

### 多重高斯模型的边缘分布与条件分布

在描述因子分析模型之前，我们先了解如何从联合多元高斯分布中找到随机变量的条件分布和边缘分布。

假设我们有一个向量值的随机变量：

<div class="math">
$$
x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix},
$$
</div>

其中 <span class="math">$x_1 \in \mathbb{R}^{r}$</span>，<span class="math">$x_2 \in \mathbb{R}^{s}$</span>，并且 <span class="math">$x \in \mathbb{R}^{r+s}$</span>。假设 <span class="math">$x \sim \mathcal{N}(\mu, \Sigma)$</span>，其中

<div class="math">
$$
\mu = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}, \quad \Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}
$$
</div>

这里，<span class="math">$\mu_1 \in \mathbb{R}^{r}$</span>，<span class="math">$\mu_2 \in \mathbb{R}^{s}$</span>，<span class="math">$\Sigma_{11} \in \mathbb{R}^{r \times r}$</span>，<span class="math">$\Sigma_{12} \in \mathbb{R}^{r \times s}$</span> ，以此类推。注意，由于协方差矩阵是对称的，有 <span class="math">$\Sigma_{12} = \Sigma_{21}^T$</span>。

根据我们的假设，<span class="math">$x_1$</span> 和 <span class="math">$x_2$</span> 是联合多元高斯分布。那么 <span class="math">$x_1$</span> 的边缘分布是什么？可以很容易地得出 <span class="math">$x_1$</span> 的期望为 <span class="math">$\text{E}[x_1] = \mu_1$</span>，协方差为 <span class="math">$\text{Cov}(x_1) = \text{E}[(x_1 - \mu_1)(x_1 - \mu_1)] = \Sigma_{11}$</span>。为了验证后面这一项成立，我们需要引入 <span class="math">$x_1$</span> 和 <span class="math">$x_2$</span> 的联合方差：

<div class="math">
$$
\begin{aligned}
\text{Cov}(x) &= \Sigma \\[5pt]
&= \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix} \\[5pt]
&= \text{E}\left[ (x - \mu)(x - \mu)^T \right] \\[5pt]
&= \text{E} \left[\begin{pmatrix} x_1 - \mu_1 \\ x_2 - \mu_2 \end{pmatrix} \begin{pmatrix} x_1 - \mu_1 \\ x_2 - \mu_2 \end{pmatrix}^T \right] \\[5pt]
&= \text{E} \begin{bmatrix} (x_1 - \mu_1)(x_1 - \mu_1)^T & (x_1 - \mu_1)(x_2 - \mu_2)^T \\ (x_2 - \mu_2)(x_1 - \mu_1)^T & (x_2 - \mu_2)(x_2 - \mu_2)^T \end{bmatrix}
\end{aligned}
$$
</div>

在上述公式的最后两行中，将矩阵中的左上方子阵匹配就能得到之前的结果。

由于高斯分布的边缘分布本身就是高斯分布，因此我们给出一个正态分布来作为 <span class="math">$x_1$</span> 的边缘分布：

<div class="math">
$$
x_1 \sim \mathcal{N}(\mu_1, \Sigma_{11}).
$$
</div>

此外，我们还可以提出另一个问题：在给定 <span class="math">$x_2$</span> 的情况下，<span class="math">$x_1$</span> 的条件分布是什么？通过多元高斯分布的定义，可以推导出：

<div class="math">
$$
x_1 | x_2 \sim \mathcal{N}(\mu_{1|2}, \Sigma_{1|2})
$$
</div>

其中：

<div class="math">
$$
\begin{align*}
\mu_{1|2} &= \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (x_2 - \mu_2) \tag{1} \\[5pt]
\Sigma_{1|2} &= \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21} \tag{2}
\end{align*}
$$
</div>

当我们在下一节中讨论因子分析模型时，这些公式对于寻找高斯分布的条件和边缘分布将起到很大的作用。

### 因子分析模型

在因子分析模型中，我们对 <span class="math">$ (x, z) $</span> 进行联合分布建模，具体如下，其中 <span class="math">$ z \in \mathbb{R}^k $</span> 是一个潜在的随机变量：

<div class="math">
$$
\begin{aligned}
z &\sim \mathcal{N}(0, I) \\[5pt]
x|z &\sim \mathcal{N}(\mu + \Lambda z, \Psi)
\end{aligned}
$$
</div>

这里，模型的参数为向量 <span class="math">$ \mu \in \mathbb{R}^n $</span>，矩阵 <span class="math">$ \Lambda \in \mathbb{R}^{n \times k} $</span>，以及对角矩阵 <span class="math">$ \Psi \in \mathbb{R}^{n \times n} $</span>。通常，<span class="math">$ k $</span> 的取值比 <span class="math">$ n $</span> 小。

因此，我们设想每个数据点 <span class="math">$ x^{(i)} $</span> 是通过从 <span class="math">$ k $</span> 维多元高斯 <span class="math">$ z^{(i)} $</span> 进行采样生成的。接着，通过计算 <span class="math">$ \mu + \Lambda z^{(i)} $</span>，它被映射到 <span class="math">$ \mathbb{R}^n $</span> 的 <span class="math">$ k $</span> 维仿射空间。最后，通过添加协方差为 <span class="math">$ \Psi $</span> 的噪声生成 <span class="math">$ x^{(i)} $</span>，即 <span class="math">$ \mu + \Lambda z^{(i)} $</span>。

我们可以定义因子分析模型如下：

<div class="math">
$$
\begin{aligned}
z &\sim \mathcal{N}(0, I) \\[5pt]
\epsilon &\sim \mathcal{N}(0, \Psi) \\[5pt]
x &= \mu + \Lambda z + \epsilon
\end{aligned}
$$
</div>

其中，<span class="math">$ \epsilon $</span> 和 <span class="math">$ z $</span> 是相互独立的。

接下来，我们将精确推导出模型定义的分布。我们的随机变量 <span class="math">$ z $</span> 和 <span class="math">$ x $</span> 具有一个联合高斯分布：

<div class="math">
$$
\begin{bmatrix} z \\ x \end{bmatrix} \sim \mathcal{N}(\mu_{zx}, \Sigma)
$$
</div>

我们现在来求 <span class="math">$ \mu_{zx} $</span> 和 <span class="math">$ \Sigma $</span>。

我们知道 <span class="math">$ \text{E}[z] = 0 $</span>，因为 <span class="math">$ z \sim \mathcal{N}(0, I) $</span>。同样地，我们也有：

<div class="math">
$$
\begin{aligned}
\text{E}[x] &= \text{E}[\mu + \Lambda z + \epsilon] \\[5pt]
&= \mu + \Lambda \text{E}[z] + \text{E}[\epsilon] \\[5pt]
&= \mu
\end{aligned}
$$
</div>

结合这些结果，我们可以得到：

<div class="math">
$$
\mu_{zx} = \begin{bmatrix} \vec{0} \\ \mu \end{bmatrix}
$$
</div>

接下来，为了求 <span class="math">$ \Sigma $</span>，我们需要计算 <span class="math">$ \Sigma_{zz} = \text{E}[(z - \text{E}[z])(z - \text{E}[z])^T] $</span>（即 <span class="math">$ \Sigma $</span> 的左上块），<span class="math">$ \Sigma_{zx} = \text{E}[(z - \text{E}[z])(x - \text{E}[x])^T] $</span>（右上块），以及 <span class="math">$ \Sigma_{xx} = \text{E}[(x - \text{E}[x])(x - \text{E}[x])^T] $</span>（右下块）。

现在，由于 <span class="math">$ z \sim \mathcal{N}(0, I) $</span>，我们很容易得出 <span class="math">$ \Sigma_{zz} = \text{Cov}(z) = I $</span>。同样地，

<div class="math">
$$
\begin{aligned}
\text{E}[(z - \text{E}[z])(x - \text{E}[x])^T] &= \text{E}[z(\mu + \Lambda z + \epsilon - \mu)^T] \\[5pt]
&= \text{E}[z z^T \Lambda^T] + \text{E}[z \epsilon^T] \\[5pt]
&= \Lambda^T
\end{aligned}
$$
</div>

在最后一步中，我们使用了 <span class="math">$ \text{E}[z z^T] = \text{Cov}(z) $</span>（因为  <span class="math">$ z $</span> 是均值为零），并且 <span class="math">$ \text{E}[z\epsilon^T] = \text{E}[z][\epsilon^T] = 0 $</span>（因为 <span class="math">$ z $</span> 和 <span class="math">$ \epsilon $</span> 相互独立）。

同理，我们可以得到 <span class="math">$ \Sigma_{xx} $</span> 如下：

<div class="math">
$$
\begin{aligned}
\text{E}[(x - \text{E}[x])(x - \text{E}[x])^T] &= \text{E}[(\mu + \Lambda z + \epsilon - \mu)(\mu + \Lambda z + \epsilon - \mu)^T] \\[5pt]
&= \text{E}[\Lambda z z^T \Lambda^T + \Lambda z \epsilon^T + \epsilon z^T \Lambda^T + \epsilon \epsilon^T] \\[5pt]
&= \Lambda \text{E}[z z^T] \Lambda^T + \text{E}[\epsilon \epsilon^T] \\[5pt]
&= \Lambda I \Lambda^T + \Psi \\[5pt]
&= \Lambda \Lambda^T + \Psi
\end{aligned}
$$
</div>

将所有结果整合在一起，我们得到：

<div class="math">
$$
\begin{bmatrix} z \\ x \end{bmatrix} \sim \mathcal{N} \left( \begin{bmatrix} \vec{0} \\ \mu \end{bmatrix}, \begin{bmatrix} I & \Lambda^T \\ \Lambda & \Lambda \Lambda^T + \Psi \end{bmatrix} \right) \tag{3}
$$
</div>

因此，我们还看到，<span class="math">$ x $</span> 的边缘分布为 <span class="math">$ z \sim \mathcal{N}(\mu, \Lambda \Lambda^T + \Psi) $</span>。于是，给定训练集 <span class="math">$ \\{x^{(i)}; i = 1, \dots, m\\} $</span>，我们可以写出参数的对数似然估计：

<div class="math">
$$
\ell(\mu, \Lambda, \Psi) = \log \prod_{i=1}^{m} \frac{1}{(2 \pi)^{n/2} |\Lambda \Lambda^T + \Psi|^{1/2}} \exp \left( - \frac{1}{2} (x^{(i)} - \mu)^T (\Lambda \Lambda^T + \Psi)^{-1} (x^{(i)} - \mu) \right)
$$
</div>

为了执行最大似然估计，我们希望最大化该函数以得到参数的值。但是显式地最大化这个公式是困难的，并且我们知道没有现成的算法能在封闭形式下解决它。因此，我们将改为使用 EM 算法。在下一节中，我们将推导因子分析的 EM 算法。

### 适用于因子分析的 EM 算法

<span class="math">$\text{E-Step}$</span> 的推导很简单。我们需要计算 <span class="math">$ Q_i(z^{(i)}) = p(z^{(i)}|x^{(i)}; \mu, \Lambda, \Psi) $</span>。通过将公式 <span class="math">$(3)$</span> 中给出的分布代入用于求高斯分布条件分布的公式 <span class="math">$(1)-(2)$</span>，我们得到：

<div class="math">
$$
z^{(i)} | x^{(i)} ; \mu, \Lambda, \Psi \sim \mathcal{N}(\mu_{z^{(i)}|x^{(i)}}, \Sigma_{z^{(i)}|x^{(i)}})
$$
</div>

其中:

<div class="math">
$$
\begin{aligned}
\mu_{z^{(i)}|x^{(i)}} &= \Lambda^T(\Lambda \Lambda^T + \Psi)^{-1}(x^{(i)} - \mu) \\[5pt]
\Sigma_{z^{(i)}|x^{(i)}} &= I - \Lambda^T(\Lambda \Lambda^T + \Psi)^{-1}\Lambda
\end{aligned}
$$
</div>

因此，通过定义 <span class="math">$ \mu_{z^{(i)}|x^{(i)}} $</span> 和 <span class="math">$ \Sigma_{z^{(i)}|x^{(i)}} $</span>，我们有：

<div class="math">
$$
Q_i(z^{(i)}) = \frac{1}{(2\pi)^{k/2} |\Sigma_{z|x}(i)|^{1/2}} \exp \left( - \frac{1}{2} (z^{(i)} - \mu_{z|x}(i))^T \Sigma_{z|x}(i)^{-1} (z^{(i)} - \mu_{z|x}(i)) \right)
$$
</div>

我们现在处理 <span class="math">$\text{M-Step}$</span>，在这里我们需要最大化：

<div class="math">
$$
\sum_{i=1}^{m} \int_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \mu, \Lambda, \Psi)}{Q_i(z^{(i)})} dz^{(i)} \tag{4}
$$
</div>

关于参数 <span class="math">$ \mu $</span>、<span class="math">$ \Lambda $</span> 和 <span class="math">$ \Psi $</span>。我们将只推导 <span class="math">$ \Lambda $</span> 的优化部分，而将 <span class="math">$ \mu $</span> 和 <span class="math">$ \Psi $</span> 的推导留给读者作为练习。

我们可以将公式<span class="math">$ (4) $</span>简化为：

<div class="math">
$$
\sum_{i=1}^{m} \int_{z^{(i)}} Q_i(z^{(i)}) \left[ \log p(x^{(i)} | z^{(i)}; \mu, \Lambda, \Psi) + \log p(z^{(i)}) - \log Q_i(z^{(i)}) \right] dz^{(i)} \tag{5}
$$
</div>

<div class="math">
$$
= \sum_{i=1}^{m} \text{E}_{z^{(i)} \sim Q_i} \left[ \log p(x^{(i)}| z^{(i)}; \mu, \Lambda, \Psi) + \log p(z^{(i)}) - \log Q_i(z^{(i)}) \right] \tag{6}
$$
</div>

其中，<span class="math">$ z^{(i)} \sim Q_i $</span> 的下标表示期望是关于从 <span class="math">$ Q_i $</span> 取得的 <span class="math">$ z^{(i)} $</span>。在随后的推导中，我们将省略此下标，以避免混淆。省略掉与参数无关的项后，我们需要最大化：

<div class="math">
$$
\begin{aligned}
\sum_{i=1}^{m} &\text{E} \left[ \log p(x^{(i)} | z^{(i)}; \mu, \Lambda, \Psi) \right] \\[5pt]
&= \sum_{i=1}^{m} \text{E} \left[ \log \frac{1}{(2\pi)^{n/2} |\Psi|^{1/2}} \exp \left( -\frac{1}{2} (x^{(i)} - \mu - \Lambda z^{(i)})^T \Psi^{-1} (x^{(i)} - \mu - \Lambda z^{(i)}) \right) \right] \\[5pt]
&= \sum_{i=1}^{m} \text{E} \left[ -\frac{1}{2} \log |\Psi| - \frac{n}{2} \log (2\pi) - \frac{1}{2} (x^{(i)} - \mu - \Lambda z^{(i)})^T \Psi^{-1} (x^{(i)} - \mu - \Lambda z^{(i)}) \right]
\end{aligned}
$$
</div>

接下来我们最大化 <span class="math">$ \Lambda $</span>。只有最后一项依赖于 <span class="math">$ \Lambda $</span>。我们对 <span class="math">$ \Lambda $</span> 取导数，得到：

<div class="math">
$$
\begin{aligned}
\nabla_{\Lambda} &\sum_{i=1}^{m} - \mathbb{E} \left[ \frac{1}{2} (x^{(i)} - \mu - \Lambda z^{(i)})^T \Psi^{-1} (x^{(i)} - \mu - \Lambda z^{(i)}) \right] \\[5pt]
&= \sum_{i=1}^{m} \nabla_{\Lambda} \mathbb{E} \left[ - \text{tr} \left( \frac{1}{2} z^{(i)^T} \Lambda^T \Psi^{-1} \Lambda z^{(i)} \right) + \text{tr} \left( z^{(i)^T} \Lambda^T \Psi^{-1} (x^{(i)} - \mu) \right) \right] \\[5pt]
&= \sum_{i=1}^{m} \nabla_{\Lambda} \mathbb{E} \left[ - \text{tr} \left( \frac{1}{2} \Lambda^T \Psi^{-1} \Lambda z^{(i)} z^{(i)^T} \right) + \text{tr} \left( \Lambda^T \Psi^{-1} (x^{(i)} - \mu) z^{(i)^T} \right) \right] \\[5pt]
&= \sum_{i=1}^{m} \mathbb{E} \left[ - \Psi^{-1} \Lambda z^{(i)} z^{(i)^T} + \Psi^{-1} (x^{(i)} - \mu) z^{(i)^T} \right]
\end{aligned}
$$
</div>

将这个式子设为零并简化，得到：

<div class="math">
$$
\sum_{i=1}^{m} \Lambda \text{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)} z^{(i)T} \right] = \sum_{i=1}^{m} (x^{(i)} - \mu) \text{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)T} \right]
$$
</div>

因此，解出  <span class="math">$ \Lambda $</span> ，我们得到：

<div class="math">
$$
\Lambda = \left( \sum_{i=1}^{m} (x^{(i)} - \mu) \text{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)T} \right] \right) \left( \sum_{i=1}^{m} \text{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)} z^{(i)T} \right] \right)^{-1}
$$
</div>

值得注意的是，这个方程与我们为最小二乘回归推导出的正规方程之间有密切的关系：

<div class="math">
$$
\theta^T = (y^T X)(X^T X)^{-1}
$$
</div>

类似之处在于，这里的 <span class="math">$ x $</span> 是关于 <span class="math">$ z $</span> 的线性函数（加上了噪声）。由于 <span class="math">$ \text{E-Step} $</span> 已经为 <span class="math">$ z $</span> 提供了"猜测"，我们现在将尝试估计未知的线性关系 <span class="math">$ \Lambda $</span> ，其将  <span class="math">$ x $</span> 与 <span class="math">$ z $</span> 联系起来。因此，出现与正规方程相似的结果并不意外。

然而，有一个重要的区别，这与只使用 <span class="math">$ z $</span> 的"最佳猜测"来执行最小二乘法的算法不同；稍后我们将看到这个差异。

为了完成我们的 <span class="math">$\text{M-Step}$</span> 更新，我们将计算方程<span class="math">$ (7) $</span>中期望的值。由于 <span class="math">$ Q_i $</span> 是高斯分布，其均值为 <span class="math">$ \mu_{z^{(i)}|x^{(i)}} $</span> 且协方差为 <span class="math">$ \Sigma_{z^{(i)}|x^{(i)}} $</span>，我们很容易得到：

<div class="math">
$$
\begin{aligned}
\text{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)} \right] = \mu_{z|x}(i) \\[5pt]
\text{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)} z^{(i)T} \right] = \mu_{z|x}(i) \mu_{z|x}(i)^T + \Sigma_{z|x}(i)
\end{aligned}
$$
</div>

后者来自于这样一个事实：对于一个随机变量 <span class="math">$ Y $</span>，有：

<div class="math">
$$
\text{Cov}(Y) = \text{E}[YY^T] - \text{E}[Y]\text{E}[Y^T]
$$
</div>

因此 <span class="math">$ \text{E}[YY^T] = \text{Cov}(Y) + \text{E}[Y]\text{E}[Y^T] $</span>。将其代入公式<span class="math">$ (7) $</span>，我们得到 <span class="math">$ \Lambda $</span> 的 <span class="math">$\text{M-Step}$</span> 更新规则：

<div class="math">
$$
\Lambda = \left( \sum_{i=1}^{m} (x^{(i)} - \mu) \mu_{z^{(i)} | x^{(i)}}^T \right) \left( \sum_{i=1}^{m} \mu_{z^{(i)} | x^{(i)}} \mu_{z^{(i)} | x^{(i)}}^T + \Sigma_{z^{(i)} | x^{(i)}} \right)^{-1} \tag{8}
$$
</div>

值得注意的是，方程右侧的 <span class="math">$ \Sigma_{z^{(i)|x^{(i)}}} $</span> 是后验分布 <span class="math">$ p(z^{(i)}|x^{(i)}) $</span> 中的协方差， <span class="math">$\text{M-Step}$</span> 必须考虑到后验中 <span class="math">$ z $</span> 的不确定性。

关于 <span class="math">$ z^{(i)} $</span> 在后验分布中的推导。一个常见的错误是在推导 EM 算法时假设在 <span class="math">$ \text{E-Step} $</span> 中只需要计算潜在随机变量 <span class="math">$ z $</span> 的期望 <span class="math">$ \text{E}[z] $</span>，然后将其代入到 <span class="math">$\text{M-Step}$</span> 的优化中。虽然这种方法适用于像高斯混合模型这样的简单问题，但在我们的因子分析推导中，我们需要 <span class="math">$ \text{E}[z z^T] $</span> 以及 <span class="math">$ \text{E}[z] $</span>。正如我们所见，<span class="math">$ \text{E}[z z^T] $</span> 和 <span class="math">$ \text{E}[z] \text{E}[z^T] $</span> 在 <span class="math">$ \Sigma_{z|x} $</span> 上有所不同。因此， <span class="math">$\text{M-Step}$</span> 更新必须考虑到在后验分布 <span class="math">$ p(z^{(i)}|x^{(i)}) $</span> 中 <span class="math">$ z $</span> 的协方差。

最后，我们还可以找到参数 <span class="math">$ \mu $</span> 和  <span class="math">$ \Psi $</span> 的 <span class="math">$\text{M-Step}$</span> 优化。由第一个公式容易得出：

<div class="math">
$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}
$$
</div>

由于这个值在参数变化时不变（即，与 <span class="math">$ \Lambda $</span> 的更新不同，右侧不依赖于 <span class="math">$ Q_i(z^{(i)}) = p(z^{(i)}|x^{(i)}; \mu, \Lambda, \Psi) $</span>，该值仅依赖于参数），因此这只需计算一次，在算法运行过程中无需进一步更新。同样，矩阵 <span class="math">$ \Psi $</span> 的对角项可以通过以下公式计算：

<div class="math">
$$
\Phi = \frac{1}{m} \sum_{i=1}^{m} \left( x^{(i)} - \mu \right) \left( x^{(i)} - \mu \right)^T - \Lambda \mu_{z|x}(i) \left( x^{(i)} - \mu \right)^T
$$
</div>

<div class="math">
$$
- \Lambda \mu_{z^{(i)}|x^{(i)}} \mu_{z^{(i)}|x^{(i)}}^T \Lambda^T + \Lambda \left( \mu_{z^{(i)}|x^{(i)}} \mu_{z^{(i)}|x^{(i)}}^T + \Sigma_{z^{(i)}|x^{(i)}} \right) \Lambda^T
$$
</div>

并将 <span class="math">$ \Psi_{ii} = \Phi_{ii} $</span> 设定为 <span class="math">$ \Phi $</span> 的对角项（即让 <span class="math">$ \Psi $</span> 为仅包含  <span class="math">$ \Phi $</span> 对角线元素的对角矩阵）。
