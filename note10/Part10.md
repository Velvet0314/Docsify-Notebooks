# 第十章&ensp;因子分析

在这一个章节里，我们会学习到如何去测试训练的模型，以及去修正模型的误差。

通过下面的学习，应该重点掌握：

* 交叉验证
* 正则化

- - -

当我们有数据  <span class="math">$ x^{(i)} \in \mathbb{R}^n $</span>  ，来自于若干个混合高斯分布时，EM 算法可以用于拟合高斯混合模型。在这种情况下，我们通常设想问题是数据足够大，以至于能够识别出数据中的多重高斯结构。例如，当训练集大小  <span class="math">$ m $</span>  明显大于数据的维度  <span class="math">$ n $</span>  时，就是这种情况。

现在，考虑一种  <span class="math">$ m \gg n $</span>  的情形。在这种情况下，用单一的高斯分布拟合数据就变得困难了，更不用说是多个高斯分布的混合了。具体而言，由于数据点基本位于  <span class="math">$ \mathbb{R}^n $</span>  的一个低维子空间中，如果我们将数据建模为高斯分布，并使用常见的最大似然估计方法来估计均值和协方差矩阵：

<div class="math">
$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}
$$
</div>

<div class="math">
$$
\Sigma = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)(x^{(i)} - \mu)^T
$$
</div>

我们会发现矩阵  <span class="math">$ \Sigma $</span>  是奇异的。这意味着  <span class="math">$ \Sigma^{-1} $</span>  不存在，并且  <span class="math">$ 1/|\Sigma|^{1/2} = 1/0 $</span> 。然而，这两个术语在计算多元高斯分布的似然时是必须的。另一种陈述此问题的方式是，最大似然估计的参数结果是一个将其所有概率集中在数据1的仿射空间中的高斯分布，并且这对应于一个奇异的协方差矩阵。

一般来说，除非  <span class="math">$ m $</span>  超过  <span class="math">$ n $</span>  达到一定的量，否则均值和协方差的最大似然估计可能会相当差。然而，我们仍然希望能够拟合一个合理的高斯模型到数据上，并可能捕捉数据中某些有趣的协方差结构。我们该如何做到这一点？

在接下来的部分中，我们首先回顾了对  <span class="math">$ \Sigma $</span>  的两种可能的限制，这些限制允许我们在少量数据下拟合  <span class="math">$ \Sigma $</span> ，但都不能给出令人满意的解决方案。然后，我们讨论了高斯分布的一些特性，这些特性将在后面需要到；特别是，如何找到高斯分布的边缘和条件分布。最后，我们呈现因子分析模型及其对应的EM算法。

### <span class="math">$\Large{\Sigma}$</span> 的约束条件

如果我们没有足够的数据来拟合一个完整的协方差矩阵，我们可以对将要考虑的矩阵  <span class="math">$\Sigma$</span>  的空间施加一些限制。例如，我们可以选择拟合一个对角线上的协方差矩阵  <span class="math">$\Sigma$</span> 。在这种情况下，读者可以很容易地验证，对角线矩阵  <span class="math">$\Sigma$</span>  的最大似然估计由以下公式给出：

<div class="math">
$$
\Sigma_{jj} = \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2
$$
</div>

因此， <span class="math">$\Sigma_{jj}$</span>  只是数据第  <span class="math">$j$</span>  个坐标的方差的经验估计值。

回顾一下，高斯密度的轮廓是椭圆形的。一个对角线的  <span class="math">$\Sigma$</span>  对应于一个椭圆的主轴不对齐的高斯分布。

有时，我们可能会对协方差矩阵施加进一步的限制，不仅要求其为对角线矩阵，而且要求其对角线元素相等。在这种情况下，我们有  <span class="math">$\Sigma = \sigma^2 I$</span> ，其中  <span class="math">$\sigma^2$</span>  是我们控制的参数。最大似然估计的  <span class="math">$\sigma^2$</span>  可以通过以下公式得到：

<div class="math">
$$
\sigma^2 = \frac{1}{mn} \sum_{j=1}^{n} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2
$$
</div>

该模型对应于使用高斯分布，其密度轮廓为圆形（在二维中）；或者在更高维度中是球形或超球形。

如果我们为数据拟合一个完整的、无约束的协方差矩阵  <span class="math">$\Sigma$</span> ，则要求  <span class="math">$m \geq n + 1$</span> ，以确保  <span class="math">$\Sigma$</span>  的最大似然估计不为奇异矩阵。在上述两种限制条件下，我们可以在  <span class="math">$m \geq 2$</span>  时得到非奇异的  <span class="math">$\Sigma$</span> 。

然而，将  <span class="math">$\Sigma$</span>  限制为对角矩阵也意味着将数据的不同坐标  <span class="math">$x_i, x_j$</span>  模型化为不相关且独立的。然而，我们通常希望能够捕捉到数据中一些有趣的相关结构。如果我们使用上述对  <span class="math">$\Sigma$</span>  的任何限制，我们将无法做到这一点。在这组笔记中，我们将描述因子分析模型，该模型使用比对角线  <span class="math">$\Sigma$</span>  更多的参数，并捕捉到数据中的一些相关性，但无需拟合完整的协方差矩阵。

### 多重高斯模型的边缘分布与条件分布

在描述因子分析模型之前，我们先来讨论如何从联合多元高斯分布中找到随机变量的条件分布和边缘分布。

假设我们有一个向量值的随机变量：

<div class="math">
$$
x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix},
$$
</div>

其中  <span class="math">$x_1 \in \mathbb{R}^{r_1}$</span> ， <span class="math">$x_2 \in \mathbb{R}^{r_2}$</span> ，并且  <span class="math">$x \in \mathbb{R}^{r}$</span> 。假设  <span class="math">$x \sim \mathcal{N}(\mu, \Sigma)$</span> ，其中

<div class="math">
$$
\mu = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}, \quad \Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}.
$$
</div>

这里， <span class="math">$\mu_1 \in \mathbb{R}^{r_1}$</span> ， <span class="math">$\mu_2 \in \mathbb{R}^{r_2}$</span> ， <span class="math">$\Sigma_{11} \in \mathbb{R}^{r_1 \times r_1}$</span> ， <span class="math">$\Sigma_{12} \in \mathbb{R}^{r_1 \times r_2}$</span> ， <span class="math">$\Sigma_{21} \in \mathbb{R}^{r_2 \times r_1}$</span> ， <span class="math">$\Sigma_{22} \in \mathbb{R}^{r_2 \times r_2}$</span> ，等等。注意，由于协方差矩阵是对称的， <span class="math">$\Sigma_{21} = \Sigma_{12}^T$</span> 。

根据我们的假设， <span class="math">$x_1$</span>  和  <span class="math">$x_2$</span>  是联合多元高斯分布。 <span class="math">$x_1$</span>  的边缘分布是什么？可以很容易地看出  <span class="math">$\mathbb{E}[x_1] = \mu_1$</span> ，并且

<div class="math">
$$
\text{Cov}(x) = \Sigma
$$
</div>

<div class="math">
$$
= \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}
$$
</div>

<div class="math">
$$
= \mathbb{E}\left[ (x - \mu)(x - \mu)^T \right]
$$
</div>

<div class="math">
$$
= \mathbb{E} \left[ \begin{bmatrix} (x_1 - \mu_1) \\ (x_2 - \mu_2) \end{bmatrix} \begin{bmatrix} (x_1 - \mu_1)^T & (x_2 - \mu_2)^T \end{bmatrix} \right]
$$
</div>

<div class="math">
$$
= \mathbb{E} \left[ \begin{bmatrix} (x_1 - \mu_1)(x_1 - \mu_1)^T & (x_1 - \mu_1)(x_2 - \mu_2)^T \\ (x_2 - \mu_2)(x_1 - \mu_1)^T & (x_2 - \mu_2)(x_2 - \mu_2)^T \end{bmatrix} \right].
$$
</div>

通过将矩阵中的左上角子块匹配，最后一行给出了结果。

由于高斯分布的边缘分布本身就是高斯分布，因此我们有：

<div class="math">
$$
x_1 \sim \mathcal{N}(\mu_1, \Sigma_{11}).
$$
</div>

此外，我们还可以问，在给定  <span class="math">$x_2$</span>  的情况下， <span class="math">$x_1$</span>  的条件分布是什么？通过多元高斯分布的定义，可以推导出

<div class="math">
$$
x_1 | x_2 \sim \mathcal{N}(\mu_{1|2}, \Sigma_{1|2}),
$$
</div>

其中

<div class="math">
$$
\mu_{1|2} = \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (x_2 - \mu_2), \tag{1}
$$
</div>

<div class="math">
$$
\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}. \tag{2}
$$
</div>

当我们在下一节中讨论因子分析模型时，这些用于寻找高斯分布的条件和边缘分布的公式将非常有用。

### 因子分析模型

在因子分析模型中，我们对  <span class="math">$ (x, z) $</span>  进行联合分布建模，具体如下，其中  <span class="math">$ z \in \mathbb{R}^k $</span>  是一个潜在的随机变量：

<div class="math">
$$
z \sim \mathcal{N}(0, I)
$$
</div>

<div class="math">
$$
x|z \sim \mathcal{N}(\mu + \Lambda z, \Psi)
$$
</div>

这里，模型的参数为向量  <span class="math">$ \mu \in \mathbb{R}^n $</span> ，矩阵  <span class="math">$ \Lambda \in \mathbb{R}^{n \times k} $</span> ，以及对角矩阵  <span class="math">$ \Psi \in \mathbb{R}^{n \times n} $</span> 。通常， <span class="math">$ k $</span>  的取值比  <span class="math">$ n $</span>  小。

因此，我们设想每个数据点  <span class="math">$ x^{(i)} $</span>  是通过从  <span class="math">$ k $</span>  维多元高斯  <span class="math">$ z^{(i)} $</span>  进行采样生成的。接着，它被映射到  <span class="math">$ \mathbb{R}^n $</span>  的  <span class="math">$ k $</span>  维仿射空间，通过计算  <span class="math">$ \mu + \Lambda z^{(i)} $</span> 。最后，通过添加协方差为  <span class="math">$ \Psi $</span>  的噪声生成  <span class="math">$ x^{(i)} $</span> ，即  <span class="math">$ \mu + \Lambda z^{(i)} $</span> 。

同样地（自己说服自己这确实是正确的），我们可以定义因子分析模型如下：

<div class="math">
$$
z \sim \mathcal{N}(0, I)
$$
</div>
<div class="math">
$$
\epsilon \sim \mathcal{N}(0, \Psi)
$$
</div>
<div class="math">
$$
x = \mu + \Lambda z + \epsilon
$$
</div>

其中， <span class="math">$ \epsilon $</span>  和  <span class="math">$ z $</span>  是相互独立的。

接下来，我们将精确推导出模型定义的分布。我们的随机变量  <span class="math">$ z $</span>  和  <span class="math">$ x $</span>  具有联合高斯分布：

<div class="math">
$$
\begin{bmatrix} z \\ x \end{bmatrix} \sim \mathcal{N}(\mu_{zx}, \Sigma).
$$
</div>

我们现在来求  <span class="math">$ \mu_{zx} $</span>  和  <span class="math">$ \Sigma $</span> 。

我们知道  <span class="math">$ \mathbb{E}[z] = 0 $</span> ，因为  <span class="math">$ z \sim \mathcal{N}(0, I) $</span> 。同样地，我们也有

<div class="math">
$$
\mathbb{E}[x] = \mathbb{E}[\mu + \Lambda z + \epsilon] = \mu + \Lambda \mathbb{E}[z] + \mathbb{E}[\epsilon] = \mu.
$$
</div>

结合这些结果，我们可以得到

<div class="math">
$$
\mu_{zx} = \begin{bmatrix} 0 \\ \mu \end{bmatrix}.
$$
</div>

接下来，为了求  <span class="math">$ \Sigma $</span> ，我们需要计算  <span class="math">$ \Sigma_{zx} = \mathbb{E}[(z - \mathbb{E}[z])(x - \mathbb{E}[x])^T] $</span>  （即  <span class="math">$ \Sigma $</span>  的左上块）， <span class="math">$ \Sigma_{zz} = \mathbb{E}[(z - \mathbb{E}[z])(z - \mathbb{E}[z])^T] $</span> （右上块），以及  <span class="math">$ \Sigma_{xx} = \mathbb{E}[(x - \mathbb{E}[x])(x - \mathbb{E}[x])^T] $</span> （右下块）。

现在，由于  <span class="math">$ z \sim \mathcal{N}(0, I) $</span> ，我们很容易得出  <span class="math">$ \Sigma_{zz} = \text{Cov}(z) = I $</span> 。同样地，

<div class="math">
$$
\mathbb{E}[(z - \mathbb{E}[z])(x - \mathbb{E}[x])^T] = \mathbb{E}[z(\mu + \Lambda z + \epsilon - \mu)^T] = \mathbb{E}[z z^T \Lambda^T + z \epsilon^T].
$$
</div>

由于  <span class="math">$ \mathbb{E}[z \epsilon^T] = 0 $</span>  （因为  <span class="math">$ z $</span>  和  <span class="math">$ \epsilon $</span>  独立），并且  <span class="math">$ \mathbb{E}[z z^T] = I $</span> ，我们得到

<div class="math">
$$
\Sigma_{zx} = \Lambda^T.
$$
</div>

在最后一步中，我们使用了  <span class="math">$ \mathbb{E}[z z^T] = \text{Cov}(z) $</span>  的结果（因为  <span class="math">$ z $</span>  是零均值），并且  <span class="math">$ \mathbb{E}[z] = 0 $</span> 。

同理，我们可以找到  <span class="math">$ \Sigma_{xx} $</span>  如下：

<div class="math">
$$
\mathbb{E}[(x - \mathbb{E}[x])(x - \mathbb{E}[x])^T] = \mathbb{E}[(\mu + \Lambda z + \epsilon - \mu)(\mu + \Lambda z + \epsilon - \mu)^T]
$$
</div>
<div class="math">
$$
= \mathbb{E}[\Lambda z z^T \Lambda^T + \Lambda z \epsilon^T + \epsilon z^T \Lambda^T + \epsilon \epsilon^T]
$$
</div>
<div class="math">
$$
= \Lambda \mathbb{E}[z z^T] \Lambda^T + \mathbb{E}[\epsilon \epsilon^T]
$$
</div>
<div class="math">
$$
= \Lambda I \Lambda^T + \Psi = \Lambda \Lambda^T + \Psi.
$$
</div>

将所有结果整合在一起，我们得到

<div class="math">
$$
\begin{bmatrix} z \\ x \end{bmatrix} \sim \mathcal{N} \left( \begin{bmatrix} 0 \\ \mu \end{bmatrix}, \begin{bmatrix} I & \Lambda^T \\ \Lambda & \Lambda \Lambda^T + \Psi \end{bmatrix} \right). \tag{3}
$$
</div>

因此，我们还看到， <span class="math">$ x $</span>  的边缘分布由  <span class="math">$ z \sim \mathcal{N}(\mu, \Lambda \Lambda^T + \Psi) $</span>  给出。于是，给定训练集  <span class="math">$ \{x^{(i)}; i = 1, \dots, m\} $</span> ，我们可以写出参数的对数似然：

<div class="math">
$$
\ell(\mu, \Lambda, \Psi) = \log \prod_{i=1}^{m} \frac{1}{(2 \pi)^{n/2} |\Lambda \Lambda^T + \Psi|^{1/2}} \exp \left( - \frac{1}{2} (x^{(i)} - \mu)^T (\Lambda \Lambda^T + \Psi)^{-1} (x^{(i)} - \mu) \right).
$$
</div>

为了执行最大似然估计，我们希望最大化该函数关于参数的值。但是显式地最大化这个公式是困难的（自己尝试一下），并且我们知道没有现成的算法能在封闭形式下解决它。因此，我们将改为使用EM算法。在下一节中，我们将推导因子分析的EM算法。

### 适用于因子分析的 EM 算法

以下是该图片内容的逐字翻译，尽量贴合术语：

---

E步的推导很简单。我们需要计算  <span class="math">$ Q_i(z^{(i)}) = p(z^{(i)}|x^{(i)}; \mu, \Lambda, \Psi) $</span> 。通过将公式 (3) 中给出的分布代入用于求高斯分布条件分布的公式 (1)-(2)，我们得到：

<div class="math">
$$
z^{(i)} | x^{(i)} ; \mu, \Lambda, \Psi \sim \mathcal{N}(\mu_{z|x}(i), \Sigma_{z|x}(i)),
$$
</div>

其中

<div class="math">
$$
\mu_{z|x}(i) = \Lambda^T(\Lambda \Lambda^T + \Psi)^{-1}(x^{(i)} - \mu),
$$
</div>
<div class="math">
$$
\Sigma_{z|x}(i) = I - \Lambda^T(\Lambda \Lambda^T + \Psi)^{-1}\Lambda。
$$
</div>

因此，使用这些定义  <span class="math">$ \mu_{z|x}(i) $</span>  和  <span class="math">$ \Sigma_{z|x}(i) $</span> ，我们有

<div class="math">
$$
Q_i(z^{(i)}) = \frac{1}{(2\pi)^{k/2} |\Sigma_{z|x}(i)|^{1/2}} \exp \left( - \frac{1}{2} (z^{(i)} - \mu_{z|x}(i))^T \Sigma_{z|x}(i)^{-1} (z^{(i)} - \mu_{z|x}(i)) \right).
$$
</div>

我们现在处理 M 步，在这里我们需要最大化：

<div class="math">
$$
\sum_{i=1}^{m} \int_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \mu, \Lambda, \Psi)}{Q_i(z^{(i)})} dz^{(i)} \tag{4}
$$
</div>

关于参数  <span class="math">$ \mu $</span> 、 <span class="math">$ \Lambda $</span>  和  <span class="math">$ \Psi $</span> 。我们将只推导  <span class="math">$ \Lambda $</span>  的优化部分，而将  <span class="math">$ \mu $</span>  和  <span class="math">$ \Psi $</span>  的推导留给读者作为练习。

我们可以将公式 (4) 简化为：

<div class="math">
$$
\sum_{i=1}^{m} \int_{z^{(i)}} Q_i(z^{(i)}) \left[ \log p(x^{(i)}, z^{(i)}; \mu, \Lambda, \Psi) + \log p(z^{(i)}) - \log Q_i(z^{(i)}) \right] dz^{(i)} \tag{5}
$$
</div>

<div class="math">
$$
= \sum_{i=1}^{m} \mathbb{E}_{z^{(i)} \sim Q_i} \left[ \log p(x^{(i)}, z^{(i)}; \mu, \Lambda, \Psi) + \log p(z^{(i)}) - \log Q_i(z^{(i)}) \right]. \tag{6}
$$
</div>

其中， <span class="math">$ z^{(i)} \sim Q_i $</span>  的下标表示期望是关于从  <span class="math">$ Q_i $</span>  采样的  <span class="math">$ z^{(i)} $</span> 。在随后的推导中，我们将省略此下标，以避免混淆。省略掉与参数无关的项后，我们需要最大化：

<div class="math">
$$
\sum_{i=1}^{m} \mathbb{E} \left[ \log p(x^{(i)}|z^{(i)}; \mu, \Lambda, \Psi) \right]
$$
</div>

<div class="math">
$$
= \sum_{i=1}^{m} \left[ -\frac{1}{2} \log |\Psi| - \frac{1}{2} (x^{(i)} - \mu - \Lambda z^{(i)})^T \Psi^{-1} (x^{(i)} - \mu - \Lambda z^{(i)}) \right].
$$
</div>

接下来我们最大化  <span class="math">$ \Lambda $</span> 。只有最后一项依赖于  <span class="math">$ \Lambda $</span> 。我们对  <span class="math">$ \Lambda $</span>  取导数，使用迹公式  <span class="math">$ tr(A) = a $</span> （对于  <span class="math">$ a \in \mathbb{R} $</span> ，且  <span class="math">$ trAB = trBA $</span> ），得到：

<div class="math">
$$
\nabla_\Lambda \sum_{i=1}^{m} \mathbb{E} \left[ - \frac{1}{2} (x^{(i)} - \mu - \Lambda z^{(i)})^T \Psi^{-1} (x^{(i)} - \mu - \Lambda z^{(i)}) \right]
$$
</div>

<div class="math">
$$
= \sum_{i=1}^{m} \mathbb{E} \left[ \Psi^{-1} (x^{(i)} - \mu) z^{(i)T} - \Psi^{-1} \Lambda z^{(i)} z^{(i)T} \right].
$$
</div>

将这个式子设为零并简化，得到：

<div class="math">
$$
\sum_{i=1}^{m} \Lambda \mathbb{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)} z^{(i)T} \right] = \sum_{i=1}^{m} (x^{(i)} - \mu) \mathbb{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)T} \right].
$$
</div>

因此，解出  <span class="math">$ \Lambda $</span> ，我们得到：

<div class="math">
$$
\Lambda = \left( \sum_{i=1}^{m} (x^{(i)} - \mu) \mathbb{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)T} \right] \right) \left( \sum_{i=1}^{m} \mathbb{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)} z^{(i)T} \right] \right)^{-1}.
$$
</div>

设此值为零并简化，我们得到：

<div class="math">
$$
\sum_{i=1}^{m} \Lambda \mathbb{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)} z^{(i)T} \right] = \sum_{i=1}^{m} (x^{(i)} - \mu) \mathbb{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)T} \right].
$$
</div>

因此，解出  <span class="math">$ \Lambda $</span> ，我们得到：

<div class="math">
$$
\Lambda = \left( \sum_{i=1}^{m} (x^{(i)} - \mu) \mathbb{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)T} \right] \right) \left( \sum_{i=1}^{m} \mathbb{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)} z^{(i)T} \right] \right)^{-1}. \tag{7}
$$
</div>

值得注意的是，这个方程与我们为最小二乘回归推导出的正规方程之间有密切的关系：

<div class="math">
$$
w \hat{y}^T = (y^T X)(X^T X)^{-1}.
$$
</div>

类似之处在于，这里的  <span class="math">$ x $</span>  是  <span class="math">$ z $</span>  的线性函数（加上噪声）。由于E步已经为  <span class="math">$ z $</span>  提供了“猜测”，我们现在将尝试估计未知的线性关系  <span class="math">$ \Lambda $</span> ，它将  <span class="math">$ x $</span>  与  <span class="math">$ z $</span>  联系起来。因此，出现与正规方程相似的结果并不令人惊讶。

然而，有一个重要的区别，这与只使用  <span class="math">$ z $</span>  的“最佳猜测”来执行最小二乘法的算法不同；我们将看到这个差异。

为了完成我们的M步更新，我们将计算方程 (7) 中期望的值。由于  <span class="math">$ Q_i $</span>  是高斯分布，均值为  <span class="math">$ \mu_{z|x}(i) $</span>  且协方差为  <span class="math">$ \Sigma_{z|x}(i) $</span> ，我们很容易得到：

<div class="math">
$$
\mathbb{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)} \right] = \mu_{z|x}(i),
$$
</div>

<div class="math">
$$
\mathbb{E}_{z^{(i)} \sim Q_i} \left[ z^{(i)} z^{(i)T} \right] = \mu_{z|x}(i) \mu_{z|x}(i)^T + \Sigma_{z|x}(i).
$$
</div>

后者来自于这样一个事实：对于一个随机变量  <span class="math">$ Y $</span> ，有

<div class="math">
$$
\text{Cov}(Y) = \mathbb{E}[YY^T] - \mathbb{E}[Y]\mathbb{E}[Y^T],
$$
</div>

因此  <span class="math">$ \mathbb{E}[YY^T] = \text{Cov}(Y) + \mathbb{E}[Y]\mathbb{E}[Y^T] $</span> 。将其代入公式 (7)，我们得到  <span class="math">$ \Lambda $</span>  的M步更新：

<div class="math">
$$
\Lambda = \left( \sum_{i=1}^{m} (x^{(i)} - \mu) \mu_{z|x}(i)^T \right) \left( \sum_{i=1}^{m} \mu_{z|x}(i) \mu_{z|x}(i)^T + \Sigma_{z|x}(i) \right)^{-1}. \tag{8}
$$
</div>

值得注意的是，方程右侧的  <span class="math">$ \Sigma_{z|x}(i) $</span>  是存在的。它是后验分布  <span class="math">$ p(z^{(i)}|x^{(i)}) $</span>  中的协方差，M步必须考虑到后验中  <span class="math">$ z $</span>  的不确定性。

关于  <span class="math">$ z^{(i)} $</span>  在后验分布中的推导。一个常见的错误是在推导EM算法时假设在E步中只需要计算潜在随机变量  <span class="math">$ z $</span>  的期望  <span class="math">$ \mathbb{E}[z] $</span> ，然后将其代入到M步的优化中。虽然这种方法适用于像高斯混合模型这样的简单问题，但在我们的因子分析推导中，我们需要  <span class="math">$ \mathbb{E}[z z^T] $</span>  以及  <span class="math">$ \mathbb{E}[z] $</span> 。正如我们所见， <span class="math">$ \mathbb{E}[z z^T] $</span>  和  <span class="math">$ \mathbb{E}[z] \mathbb{E}[z^T] $</span>  在量  <span class="math">$ \Sigma_{z|x} $</span>  上有所不同。因此，M步更新必须考虑到在后验分布  <span class="math">$ p(z^{(i)}|x^{(i)}) $</span>  中  <span class="math">$ z $</span>  的协方差。

最后，我们还可以找到参数  <span class="math">$ \mu $</span>  和  <span class="math">$ \Psi $</span>  的M步优化。第一个公式很容易得出：

<div class="math">
$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}.
$$
</div>

由于这个值在参数变化时不变（即，与  <span class="math">$ \Lambda $</span>  的更新不同，右侧不依赖于  <span class="math">$ Q_i(z^{(i)}) = p(z^{(i)}|x^{(i)}; \mu, \Lambda, \Psi) $</span> ，该值仅依赖于参数），因此这只需计算一次，在算法运行过程中无需进一步更新。同样，矩阵  <span class="math">$ \Psi $</span>  的对角项可以通过以下公式计算：

<div class="math">
$$
\Phi = \frac{1}{m} \sum_{i=1}^{m} \left( x^{(i)} - \mu \right) \left( x^{(i)} - \mu \right)^T - \Lambda \mu_{z|x}(i) \left( x^{(i)} - \mu \right)^T
$$
</div>

<div class="math">
$$
- \Lambda \mu_{z|x}(i) \mu_{z|x}(i)^T \Lambda^T + \Lambda \left( \mu_{z|x}(i) \mu_{z|x}(i)^T + \Sigma_{z|x}(i) \right) \Lambda^T,
$$
</div>

并将  <span class="math">$ \Psi_{ii} = \Phi_{ii} $</span>  设定为  <span class="math">$ \Phi $</span>  的对角项（即让  <span class="math">$ \Psi $</span>  为仅包含  <span class="math">$ \Phi $</span>  对角线元素的对角矩阵）。
