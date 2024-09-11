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

### EM 算法详解

假设我们有一个估计问题，其中我们有一个训练集 <span class="math">$\\{x^{(1)}, \dots, x^{(m)}\\}$</span>，由 <span class="math">$m$</span> 个独立的样本组成。我们希望将模型 <span class="math">$p(x, z)$</span> 的参数 <span class="math">$\theta$</span> 拟合到数据上，此时似然函数为：

<div class="math">
$$
\begin{align*}
\ell(\theta) &= \sum_{i=1}^{m} \log p(x^{(i)}; \theta) \\[5pt]
&= \sum_{i=1}^{m} \log \sum_{z} p(x, z; \theta)
\end{align*}
$$
</div>

但是，显式地找到参数 <span class="math">$\theta$</span> 的最大似然估计可能会很困难。这里，<span class="math">$z^{(i)}$</span> 是潜在随机变量；而且通常情况下，如果我们能够观察到这些 <span class="math">$z^{(i)}$</span>，那么最大似然估计就会变得比较简单。

在这种情况下，EM 算法提供了一种高效的方法用来构建最大似然估计。直接最大化 <span class="math">$\ell(\theta)$</span> 可能较为困难，因此我们的策略是与之前类似：构建一个似然函数的下界（<span class="math">$ \text{E-Step} $</span>），然后优化该下界（<span class="math">$ \text{M-Step} $</span>）。

对于每一个 <span class="math">$i$</span>，令 <span class="math">$Q_i$</span> 为某种对 <span class="math">$ z $</span> 的分布（满足 <span class="math">$\sum_z Q_i(z) = 1, Q_i(z) \geq 0$</span>）。考虑如下过程：

<div class="math">
$$
\begin{align*}
\sum_i \log p(x^{(i)}; \theta) &= \sum_i \log \sum_{z^{(i)}} p(x^{(i)}, z^{(i)}; \theta) \tag{1} \\[5pt]
&= \sum_i \log \sum_{z^{(i)}} Q_i(z^{(i)}) \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \tag{2} \\[5pt]
&\geq \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \tag{3}
\end{align*}
$$
</div>

这个推导的最后一步使用了 Jensen 不等式。具体来说，<span class="math">$f(x) = \log x$</span>是一个凹函数，并且它的二阶导数 <span class="math">$f^{\prime\prime}(x) = -1/x^2 < 0$</span>，在定义域 <span class="math">$x \in \mathbb{R^+}$</span> 内成立。同时，

<div class="math">
$$
\sum_{z^{(i)}} Q_i(z^{(i)}) \left[\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}\right]
$$
</div>

在求和中是关于 <span class="math">$\left[p(x^{(i)}, z^{(i)}; \theta) / Q_i(z^{(i)})\right]$</span> 的期望，其中  <span class="math">$z^{(i)}$</span> 从 <span class="math">$Q_i$</span> 分布中抽取。通过 Jensen 不等式，我们有：

<div class="math">
$$
f\left(E_{z^{(i)} \sim Q_i} \left[\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}\right]\right) \geq E_{z^{(i)} \sim Q_i} \left[f\left(\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}\right)\right]
$$
</div>

这里，符号 <span class="math">$z^{(i)} \sim Q_i$</span> 表示期望是相对于从 <span class="math">$Q_i$</span> 中抽样的 <span class="math">$z^{(i)}$</span> 进行的。这使我们能够从等式<span class="math">$(2)$</span>推导出不等式<span class="math">$(3)$</span>。

现在，对于任意分布 <span class="math">$Q_i$</span>，公式<span class="math">$(3)$</span> 为 <span class="math">$\ell(\theta)$</span> 提供了一个下界。同时，对于 <span class="math">$Q_i$</span> 的选择存在许多可能性。那么，应该如何选择呢？假设我们有当前的参数 <span class="math">$\theta$</span> 的某种估计，那么似乎自然地，我们应该尝试在参数为 <span class="math">$\theta$</span> 的当前值上使下界尽可能地贴近。也就是说，我们将尝试使不等式在当前参数值 <span class="math">$\theta$</span> 上取等号。（稍后我们将看到这使得 <span class="math">$\ell(\theta)$</span> 在EM 算法的迭代中单调递增。）

为了使不等式在特定值 <span class="math">$\theta$</span> 上成立，我们需要在推导过程中应用 Jensen 不等式时，确保是对一个常数值随机变量求期望。也就是说，我们需要确保：

<div class="math">
$$
\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} = c
$$
</div>

对于某个常量 <span class="math">$c$</span> ，它不依赖于 <span class="math">$z^{(i)}$</span> 。这可以通过选择

<div class="math">
$$
Q_i(z^{(i)}) \propto p(x^{(i)}, z^{(i)}; \theta)
$$
</div>

轻松实现。实际上，由于我们知道 <span class="math">$\sum_z Q_i(z) = 1$</span> （因为它是一个分布），这进一步告诉我们：

<div class="math">
$$
\begin{aligned}
Q_i(z^{(i)}) &= \frac{p(x^{(i)}, z^{(i)}; \theta)}{\sum_{z^{(i)}} p(x^{(i)}, z^{(i)}; \theta)} \\[5pt]
&= \frac{p(x^{(i)}, z^{(i)}; \theta)}{p(x^{(i)}; \theta)} \\[5pt]
&= p(z^{(i)}|x^{(i)};\theta)
\end{aligned}
$$
</div>

因此， 在给定 <span class="math">$x^{(i)}$</span> 和当前参数设定 <span class="math">$\theta$</span> 的条件下， 我们只需要选择 <span class="math">$Q_i$</span> 作为 <span class="math">$z^{(i)}$</span> 的后验分布。

现在，在这种 <span class="math">$Q_i$</span> 的选择下，公式(3)给出了 <span class="math">$\theta$</span> 似然函数的下界。这就是E步。在 EM 算法的M步中，我们针对参数 <span class="math">$\theta$</span> 最大化公式(3)来获得一个新的参数设定。重复执行这两步，即得到 EM 算法，具体如下：

> [!TIP]
> <div class="math">
> $$
> \begin{array}{l}
> \text{Repeat until convergence: } \{ \\[5pt]
> \qquad \text{(E-step): For each } i, \text{ set} \\[5pt]
> \qquad \qquad \qquad Q_i(z^{(i)}) := p(z^{(i)}|x^{(i)}; \theta) \\[5pt]
> \qquad \text{(M-step): Set} \\[5pt]
> \qquad \qquad \qquad \theta := \text{argmax}_{\theta} \sum_{i} \sum_{z^{(i)}} Q_i(z^{(i)}) \log \displaystyle{\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}} \\[5pt]
> \}
> \end{array}
> $$
> </div>

如何判断该算法是否会收敛呢？我们设 <span class="math">$\theta^{(t)}$</span> 和 <span class="math">$\theta^{(t+1)}$</span> 是 EM 算法中两次迭代的参数。现在我们将证明 <span class="math">$\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$</span> ，这表明 EM 算法会总是会使似然函数单调递增。

证明该结论的关键在于我们对 <span class="math">$Q_i$</span> 的选择。具体来说，在 EM 算法的迭代中，当参数从 <span class="math">$\theta^{(t)}$</span> 开始时，我们将选择 <span class="math">$Q_i^{(t+1)}(z^{(i)}) = p(z^{(i)} | x^{(i)}; \theta^{(t)})$</span> 。我们之前已经看到，这一选择确保了 Jensen 不等式成立，并应用到公式<span class="math">$(3)$</span>，即：

<div class="math">
$$
\ell(\theta^{(t)}) = \sum_i \sum_{z^{(i)}} Q_i^{(t)}(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta^{(t)})}{Q_i^{(t)}(z^{(i)})}
$$
</div>

通过最大化右边的值，参数 <span class="math">$\theta^{(t+1)}$</span> 可以通过以下方式获得：

<div class="math">
$$
\begin{align*}
\ell(\theta^{(t+1)}) &= \sum_i \sum_{z^{(i)}} Q_i^{(t)}(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta^{(t+1)})}{Q_i^{(t)}(z^{(i)})} \tag{4} \\[5pt]
&\geq \sum_i \sum_{z^{(i)}} Q_i^{(t)}(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta^{(t)})}{Q_i^{(t)}(z^{(i)})} \tag{5} \\[5pt]
&= \ell(\theta^{(t)}) \tag{6}
\end{align*}
$$
</div>

第一个不等式来自于：

<div class="math">
$$
\ell(\theta) = \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
$$
</div>

对于任意的 <span class="math">$Q_i$</span> 和 <span class="math">$\theta$</span> 都成立，特别是当 <span class="math">$Q_i = Q_i^{(t+1)}$</span> 时， <span class="math">$\theta = \theta^{(t)}$</span> 。为了从公式(5)推导出公式(6)，我们使用了 <span class="math">$\theta^{(t+1)}$</span> 的选择：

<div class="math">
$$
\theta^{(t+1)} = \arg\max_\theta \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
$$
</div>

因此，当评估 <span class="math">$\theta^{(t+1)}$</span> 时，这个公式的值必须等于或大于在 <span class="math">$\theta^{(t)}$</span> 时的值。最后，由于公式<span class="math">$(5)$</span>已经成立，因此公式<span class="math">$(6)$</span>得证，即 <span class="math">$\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$</span> ，表明 <span class="math">$\ell(\theta)$</span> 在 <span class="math">$\theta = \theta^{(t+1)}$</span> 时单调递增。

因此，EM 算法使得似然函数单调收敛。在我们对 EM 算法的描述中，提到我们会运行它直到其收敛。根据我们刚刚展示的结果，一种合理的收敛测试方法是检查相邻迭代之间的 <span class="math">$\ell(\theta)$</span> 增量是否小于某个容差参数，并且如果 EM 算法在提高 <span class="math">$(\ell(\theta)$</span> 的速度过慢时，就可以表明其收敛。

> [!NOTE]
> 如果我们定义
> <div class="math">
> $$
> J(Q, \theta) = \sum_i \sum_{z(i)} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})},
> $$
> </div>
>
> 那么根据我们之前的推导，我们知道 <span class="math">$\ell(\theta) \geq J(Q, \theta) $</span> 。EM算法也可以被视为在 <span class="math">$J $</span> 上的坐标上升，其中 <span class="math">$ \text{E-Step} $</span> 最大化 <span class="math">$J $</span> 的 <span class="math">$Q$</span> 部分（请自行验证），而 <span class="math">$ \text{M-Step} $</span> 最大化 <span class="math">$\theta $</span> 部分。

### 高斯混合模型详解

在我们对 EM 算法的一般定义的基础上，我们回到之前的老例子：拟合高斯混合模型的参数 <span class="math">$\phi $</span> 、 <span class="math">$\mu $</span> 和 <span class="math">$\Sigma $</span> 。为了简化推导，我们仅进行 <span class="math">$ \text{M-Step} $</span> 更新 <span class="math">$\phi $</span> 和 <span class="math">$\mu_j $</span> 的推导， <span class="math">$\Sigma_j $</span> 的更新留给读者自行推导。

<span class="math">$ \text{E-Step} $</span>相对简单。根据我们之前的推导，我们可以直接计算

<div class="math">
$$
w_j^{(i)} = Q_i(z^{(i)} = j) = P(z^{(i)} = j | x^{(i)}; \phi, \mu, \Sigma)。
$$
</div>

其中，“ <span class="math">$Q_i(z^{(i)} = j) $</span> ”表示在分布 <span class="math">$Q_i $</span> 下， <span class="math">$z^{(i)} $</span> 取值为 <span class="math">$j $</span> 的概率。

接下来，在M步中，我们需要对参数 <span class="math">$\phi $</span> 、 <span class="math">$\mu $</span> 、 <span class="math">$\Sigma $</span> 进行最大化，具体要最大化的是：

<div class="math">
$$
\begin{aligned}
\sum_{i=1}^{m} \sum_{z^{(i)}} &Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \phi, \mu, \Sigma)}{Q_i(z^{(i)})} \\[5pt]
&=\sum_{i=1}^{m} \sum_{j=1}^{k} Q_i(z^{(i)} = j) \log \frac{p(x^{(i)} | z^{(i)} = j; \mu, \Sigma)p(z^{(i)} = j; \phi)}{Q_i(z^{(i)} = j)} \\[5pt]
&=\sum_{i=1}^{m} \sum_{j=1}^{k} w_j^{(i)} \log \frac{1}{(2\pi)^{n/2} |\Sigma_j|^{1/2}} \exp \left(-\frac{1}{2}(x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j)\right) \cdot \phi_j
\end{aligned}
$$
</div>

我们对 <span class="math">$\mu_l $</span> 进行最大化。对 <span class="math">$\mu_l $</span> 求导，得到：

<div class="math">
$$
\begin{aligned}
\nabla_{\mu_l} \sum_{i=1}^{m} &\sum_{j=1}^{k} w_j^{(i)} \frac{\log \frac{1}{(2\pi)^{n/2} |\Sigma_j|^{1/2}} \exp \left(-\frac{1}{2}(x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j)\right) \cdot \phi_j}{w_j^{(i)}} \\[5pt]
&=-\nabla_{\mu_l} \sum_{i=1}^{m} \sum_{j=1}^{k} w_j^{(i)} \frac{1}{2}(x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j) \\[5pt]
&=\frac{1}{2} \sum_{i=1}^{m} w_l^{(i)} \ nabla_{\mu_l} 2 \mu_l^T \Sigma_l^{-1} x^{(i)} - \mu_l^T \Sigma_l^{-1} \mu_l \\[5pt]
&=\sum_{i=1}^{m} w_l^{(i)} (\Sigma_l^{-1} x^{(i)} - \Sigma_l^{-1} \mu_l)
\end{aligned}
$$
</div>

将此设为零并解出 <span class="math">$\mu_l $</span> ，得到更新公式：

<div class="math">
$$
\mu_l := \frac{\sum_{i=1}^{m} w_l^{(i)} x^{(i)}}{\sum_{i=1}^{m} w_l^{(i)}}
$$
</div>

这就是我们在前面推导出的结果。

再举一个例子，推导参数 <span class="math">$\phi_j $</span> 的M步更新。我们只收集依赖于 <span class="math">$\phi_j $</span> 的项，得到我们需要最大化的目标函数：

<div class="math">
$$
\sum_{i=1}^{m} \sum_{j=1}^{k} w_j^{(i)} \log \phi_j。
$$
</div>

然而，有一个额外的约束条件： <span class="math">$\phi_j $</span> 的和为1，因为 <span class="math">$\phi_j $</span> 表示 <span class="math">$p(z^{(i)} = j; \phi) $</span> 的概率。为了处理这个约束条件 <span class="math">$\sum_{j=1}^{k} \phi_j = 1 $</span> ，我们构建拉格朗日函数：

<div class="math">
$$
\mathcal{L}(\phi) = \sum_{i=1}^{m} \sum_{j=1}^{k} w_j^{(i)} \log \phi_j + \beta \left( \sum_{j=1}^{k} \phi_j - 1 \right)，
$$
</div>

其中 <span class="math">$\beta $</span> 是拉格朗日乘数。对 <span class="math">$\phi_j $</span> 求导，我们得到：

<div class="math">
$$
\frac{\partial}{\partial \phi_j} \mathcal{L}(\phi) = \sum_{i=1}^{m} \frac{w_j^{(i)}}{\phi_j} + \beta。
$$
</div>

将此设为零并解得：

<div class="math">
$$
\phi_j = \frac{\sum_{i=1}^{m} w_j^{(i)}}{-\beta}。
$$
</div>

即 <span class="math">$\phi_j \propto \sum_{i=1}^{m} w_j^{(i)} $</span> 。利用 <span class="math">$\sum_j \phi_j = 1 $</span> 的约束条件，我们很容易得到 <span class="math">$-\beta = \sum_{i=1}^{m} \sum_{j=1}^{k} w_j^{(i)} = m $</span> ，因此有：

<div class="math">
$$
\phi_j := \frac{1}{m} \sum_{i=1}^{m} w_j^{(i)}。
$$
</div>

<span class="math">$\Sigma_j $</span> 的 <span class="math">$\text{M-Step}$</span> 更新推导也是相对简单的。
