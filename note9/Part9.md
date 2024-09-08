# 第九章&ensp;EM 算法

在这一个章节里，我们会学习 EM 算法及其推广和应用。

通过下面的学习，应该重点掌握：

* 高斯混合模型
* EM 算法

- - -

### 高斯混合模型

假设我们给定一个训练集 <span class="math">$ \\{x^{(1)}, \dots, x^{(m)}\\} $</span>。由于我们假设这次在无监督学习的环境中，所以这些数据点并没有具体的标签。

我们希望通过得到一个联合分布 <span class="math">$ p(x^{(i)}, z^{(i)}) = p(x^{(i)}|z^{(i)})p(z^{(i)}) $</span> 来对数据进行建模。这里，<span class="math">$ z^{(i)} \sim \text{Multinomial}(\phi) $</span>。

<span class="math">$ z^{(i)} $</span> 是一个以 <span class="math">$ \phi $</span> 为参数的多项式分布。其中 <span class="math">$ \phi_j \geq 0 $</span>，<span class="math">$ \sum_{j=1}^k \phi_j = 1 $</span>，并且参数 <span class="math">$ \phi_j $</span> 给出了 <span class="math">$ p(z^{(i)} = j) $</span>，同时 <span class="math">$ x^{(i)}|z^{(i)} = j \sim \mathcal{N}(\mu_j, \Sigma_j) $</span>。

我们令 <span class="math">$ k $</span> 表示 <span class="math">$ z^{(i)} $</span> 可能取值的数量。因此，我们的模型假设每个 <span class="math">$ x^{(i)} $</span> 是通过从 <span class="math">$ \\{1, \dots, k\\} $</span> 中随机选择 <span class="math">$ z^{(i)} $</span> 生成的，然后 <span class="math">$ x^{(i)} $</span> 是根据所选择的 <span class="math">$ z^{(i)} $</span> 对应的高斯分布中抽取的。这就是所谓的 **高斯混合模型（mixture of Gaussians model）**。另外，注意到 <span class="math">$ z^{(i)} $</span> 是 **潜在（latent）** 随机变量，这意味着它们是隐藏或是不可观测的，并会使得我们的估计问题变得复杂。

> [!NOTE]
> 选取的分布取决于 <span class="math">$ z^{(i)} $</span> 对应的各自的高斯分布，而非简单地从一个单独的高斯分布中生成 <span class="math">$ x^{(i)} $</span>

我们模型的参数因此是 <span class="math">$ \phi $</span>、<span class="math">$ \mu $</span> 和 <span class="math">$ \Sigma $</span>。要对这些值进行估计，我们可以写出数据的似然函数：

<div class="math">
$$
\begin{aligned}
\ell(\phi, \mu, \Sigma) &= \sum_{i=1}^m \log p(x^{(i)}; \phi, \mu, \Sigma) \\[5pt]
&= \sum_{i=1}^m \log \sum_{z^{(i)}=1}^k p(x^{(i)}|z^{(i)}; \mu, \Sigma)p(z^{(i)}; \phi)
\end{aligned}
$$
</div>

然而，如果我们直接对该似然函数关于参数取导数并尝试求解，会发现根本不可能以闭合形式来找到这些参数的最大似然估计。

随机变量 <span class="math">$ z^{(i)} $</span> 指示了每个 <span class="math">$ x^{(i)} $</span> 是从哪一个高斯分布生成的。

如果 <span class="math">$ z^{(i)} $</span> 是已知的，那么最大似然问题将会非常简单。具体来说，我们可以写出似然函数为

<div class="math">
$$
\ell(\phi, \mu, \Sigma) = \sum_{i=1}^{m} \log p(x^{(i)}|z^{(i)}; \mu, \Sigma) + \log p(z^{(i)}; \phi)
$$
</div>

对函数进行求解最大化可以得到参数 <span class="math">$ \phi $</span>、<span class="math">$ \mu $</span> 和 <span class="math">$ \Sigma $</span> 的更新公式：

<div class="math">
$$
\begin{aligned}
\phi_j &= \frac{1}{m} \sum_{i=1}^{m} 1\{z^{(i)} = j\} \\[5pt]
\mu_j &= \frac{\sum_{i=1}^{m} 1\{z^{(i)} = j\}x^{(i)}}{\sum_{i=1}^{m} 1\{z^{(i)} = j\}} \\[5pt]
\Sigma_j &= \frac{\sum_{i=1}^{m} 1 \{z^{(i)} = j\}(x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_{i=1}^{m} 1 \{z^{(i)} = j\}}
\end{aligned}
$$
</div>

实际上，我们可以看到，如果 <span class="math">$ z^{(i)} $</span> 是已知的，那么最大似然估计几乎与我们在高斯判别分析模型中估计参数的过程完全相同。不同的是这里的 <span class="math">$ z^{(i)} $</span> 起到了 GDA 中类别标签的作用。

> [!NOTE]
> 这里在公式上与 GDA 的结果存在一些细微差异，首先是因为我们将 <span class="math">$ z^{(i)} $</span> 从伯努利分布推广到多项式分布，其次是因为这里我们为每个高斯分布使用了不同的协方差矩阵 <span class="math">$ \Sigma_j $</span>。

然而，在我们的密度估计问题中，<span class="math">$ z^{(i)} $</span> 是未知的。那么我们该怎么办呢？接下来我们给出 EM 算法来解决这个问题。

### EM 算法

**EM 算法（Expectation-Maximization algorithm）** 是一种迭代算法，主要包括两个步骤。对于我们的问题，<span class="math">$E$</span> 步骤尝试"猜测"（guess） <span class="math">$ z^{(i)} $</span> 的值。<span class="math">$M$</span> 步骤则基于我们的猜测更新模型的参数。在 <span class="math">$M$</span> 步骤中，假设第一步的猜测是正确的，那么最大化问题就变得简单了。算法如下：

> [!TIP]
> <div class="math">
> $$
> \begin{array}{l}
> \text{Repeat Until Convergence: } \{ \\[5pt]
> \qquad \text{(E-Step): For each } i, j, \text{ set} \\[5pt]
> \qquad \qquad \qquad w_j^{(i)} := p(z^{(i)} = j | x^{(i)}; \phi, \mu, \Sigma) \\[5pt]
> \qquad \text{(M-Step): Update the parameters:} \\[5pt]
> \qquad \qquad \qquad \displaystyle{\phi_j := \frac{1}{m} \sum_{i=1}^m w_j^{(i)}}, \\[5pt]
> \qquad \qquad \qquad \displaystyle{\mu_j := \frac{\sum_{i=1}^m w_j^{(i)} x^{(i)}}{\sum_{i=1}^m w_j^{(i)}}}, \\[5pt]
> \qquad \qquad \qquad \displaystyle{\Sigma_j := \frac{\sum_{i=1}^m w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_{i=1}^m w_j^{(i)}}}, \\[5pt]
> \}
> \end{array}
> $$
> </div>

在 <span class="math">$E$</span> 步骤中，我们根据当前参数的设定，给定数据 <span class="math">$ x^{(i)} $</span>，通过贝叶斯法则计算参数 <span class="math">$ z^{(i)} $</span> 的后验概率，我们可以得到：

<div class="math">
$$
p(z^{(i)} = j | x^{(i)}; \phi, \mu, \Sigma) = \frac{p(x^{(i)} | z^{(i)} = j; \mu, \Sigma) p(z^{(i)} = j; \phi)}{\sum_{l=1}^k p(x^{(i)} | z^{(i)} = l; \mu, \Sigma) p(z^{(i)} = l; \phi)}
$$
</div>

这里，<span class="math">$ p(x^{(i)} | z^{(i)} = j; \mu, \Sigma) $</span> 是通过评估均值为 <span class="math">$ \mu_j $</span>、协方差为 <span class="math">$ \Sigma_j $</span> 的高斯分布的密度得到的，并且 <span class="math">$ p(z^{(i)} = j; \phi) $</span> 由 <span class="math">$ \phi_j $</span> 给出，以此类推。在 <span class="math">$E$</span> 步骤中计算的 <span class="math">$ w_j^{(i)} $</span> 代表了我们对 <span class="math">$ z^{(i)} $</span> 值的"弱（soft）估计"。

> [!NOTE]
> 这里"弱"指的是对概率的猜测，从 <span class="math">$ [0,1] $</span> 这样一个闭区间取值；而"强"指的是单次最佳猜测，例如从集合 <span class="math">$ \\{ 0,1 \\}$</span> 或 <span class="math">$ \\{ 1,..,k \\}$</span> 中取一个值

同时，你应该对比一下 <span class="math">$M$</span> 步骤中的更新公式与我们之前在 <span class="math">$ z^{(i)} $</span> 已知时的更新公式。两者实际上是相同的，但这里我们用 <span class="math">$ w_j^{(i)} $</span> 代替了之前用于指示其属于哪个高斯分布的指示函数 <span class="math">$ 1\\{z^{(i)} = j\\} $</span>。

EM 算法也让我们回忆起了 K-means 聚类算法。不同的是，在这里我们用"弱"赋值 <span class="math">$ w_j^{(i)} $</span> 代替了K-means中"强"的聚类赋值 <span class="math">$ c(i) $</span>。类似于 K-means，它也容易陷入局部最优解，因此在多个不同的初始参数设置下重新初始化可能效果会更好。

很明显，EM 算法有一个非常自然的解释，即反复尝试猜测未知的 <span class="math">$ z^{(i)} $</span> 值；但是它是如何出现的呢？我们能否对其收敛性等问题做出任何保证呢？我们将在后续讨论 EM 算法的一个更通用的情况。

### Jensen 不等式

在进行 EM 算法的进一步讨论前，我们先需要了解 Jensen 不等式。

令 <span class="math">$ f $</span> 为定义在实数集上的函数。如果 <span class="math">$ f $</span> 是一个凸函数，那么对于所有 <span class="math">$ x \in \mathbb{R} $</span> 都有 <span class="math">$ f''(x) \geq 0 $</span>。对于向量值输入，这一条件可以推广为其 Hessian 矩阵 <span class="math">$ H $</span> 是半正定的（<span class="math">$ H \geq 0 $</span>）。如果对所有 <span class="math">$ x $</span> 都有 <span class="math">$ f''(x) > 0 $</span>，则称 <span class="math">$ f $</span> 是严格凸函数（在向量值情况下，条件是 Hessian 矩阵 <span class="math">$ H $</span> 必须是正定的，即 <span class="math">$ H > 0 $</span>）。Jensen 不等式可以表述如下：

> [!TIP]
> **<font size=4>Jensen 不等式</font>**
>
> **定理**：设 <span class="math">$ f $</span> 为一个凸函数，<span class="math">$ X $</span> 为随机变量，则有：
>
> <div class="math">
> $$
> E[f(X)] \geq f(E[X])
> $$
> </div>

此外，如果 <span class="math">$ f $</span> 是严格凸函数，那么 <span class="math">$ E[f(X)] = f(E[X]) $</span> 仅当 <span class="math">$ X = E[X] $</span> 必然成立（即 <span class="math">$ X $</span> 是一个常量）。

在书写期望时，我们有时会省略括号。因此，在上面的定理中，<span class="math">$ f(EX) = f(E[X]) $</span>。

为了更好地理解该定理，请参考下面的图示。

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/09/08/pAeWqSI.png" data-lightbox="image-9" data-title="Jensen inequality">
  <img src="https://s21.ax1x.com/2024/09/08/pAeWqSI.png" alt="Jensen inequality" style="width:100%;max-width:450px;cursor:pointer">
 </a>
</div>

在这里，<span class="math">$ f $</span> 是由实线表示的凸函数。另外，<span class="math">$ X $</span> 是一个随机变量，它有 <span class="math">$0.5$</span> 的概率取值为 <span class="math">$ a $</span>，另有 <span class="math">$0.5$</span> 的概率取值为 <span class="math">$ b $</span>（如图中 x 轴所示）。因此，<span class="math">$ X $</span> 的期望值 <span class="math">$ E[X] $</span> 是 <span class="math">$ a $</span> 和 <span class="math">$ b $</span> 的中点。

我们还可以看到 <span class="math">$ f(a) $</span>、<span class="math">$ f(b) $</span> 和 <span class="math">$ f(E[X]) $</span> 分别标示在 <span class="math">$y$</span> 轴上。此外，值 <span class="math">$ E[f(X)] $</span> 是 <span class="math">$ f(a) $</span> 和 <span class="math">$ f(b) $</span> 之间在 <span class="math">$y$</span> 轴上的中点。在这个例子中，由于 <span class="math">$ f $</span> 是凸的，必然有 <span class="math">$ E[f(X)] \geq f(E[X]) $</span>。

> [!NOTE]
> 如果且仅当 <span class="math">$ -f $</span> 是 严格凸函数，则 <span class="math">$ f $</span> 是严格凹函数（即 <span class="math">$ f''(x) \leq 0 $</span> 或 <span class="math">$ H \leq 0 $</span>）。Jensen 不等式也适用于凹函数 <span class="math">$ f $</span>，但是不等式的符号是反过来的。（即 <span class="math">$ E[f(X)] \leq f(E[X]) $</span>）

### EM 算法推广

2

### 高斯混合模型

3
