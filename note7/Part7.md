# 第七章&ensp;学习理论

在先前的章节里，我们基本已经对机器学习中的学习方法有了基本的了解。接下来，我们将以各种方法为引，对机器学习中的核心理论进行分析，以强化我们的理论学习，帮助我们进行误差和性能分析。

通过下面的学习，应该重点掌握：

* 偏差/方差权衡
* 切尔诺夫界
* VC 维

- - -

### 偏差/方差权衡

在讨论线性回归时，我们讨论了应该如何选择正确的模型：是应该拟合"简单"模型（例如线性模型 <span class="math"> $ y = \theta_0 + \theta_1 x $</span>），还是更"复杂"的模型（例如多项式 <span class="math">$ y = \theta_0 + \theta_1 x + \dots + \theta_5 x^5 $</span>）。我们看到以下示例：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/08/28/pAACJzT.png" data-lightbox="image-7" data-title="model choose">
  <img src="https://s21.ax1x.com/2024/08/28/pAACJzT.png" alt="model choose" style="width:100%;max-width:850px;cursor:pointer">
 </a>
</div>

上面的三张图像，分别展示了不同拟合复杂度的曲线。

拟合5次多项式到数据上（最右边的图像）并没有得到一个好的模型。具体来说，即使5次多项式对训练集中的 <span class="math">$ y $</span>（比如房价）和 <span class="math">$ x $</span>（比如居住面积）做出了非常好的预测，我们并不期望这个模型能很好地预测训练集中没有出现的房屋价格。换句话说，从训练集中学习到的内容并没有很好地 **泛化（generalize）** 到其他房屋。**泛化误差（generalization error）**（稍后将正式定义）是指那些不属于训练集的样本潜在的预期偏差。

最左边的模型和最右边的模型都存在较大的泛化误差。然而，这两个模型的问题来源却是大相径庭。如果 <span class="math">$ y $</span> 和 <span class="math">$ x $</span> 之间的关系不是线性的，即使我们将线性模型拟合到非常大量的训练数据上，线性模型仍然无法准确捕捉数据中的结构。我们将 **偏差（bias）** 定义为模型的预期泛化误差，即便我们要去拟合的对象是一个非常大的甚至是无限的训练数据集。因此，对于上述问题，线性模型存在较大的偏差，并且可能出现欠拟合（即，未能捕捉到数据中的结构特征）。

除了偏差之外，泛化误差的另一个来源是模型拟合过程中的 **方差（variance）**。具体来说，当我们像右边图中那样拟合五次多项式时，存在较大风险，即我们拟合了训练集中偶然存在的模式，而这些模式并没有反映 $ x $ 和 $ y $ 之间的更广泛关系。这可能是因为在训练集中，我们碰巧得到了稍微比平均价格更贵的房子，或者稍微比平均价格更便宜的房子，等等。通过拟合这些"虚假"的训练集模式，我们可能会得到一个具有较大的泛化误差的模型。在这种情况下，我们说模型具有较大的方差。

因此，在偏差和方差之间存在权衡。如果我们的模型太"简单"并且参数很少，那么它可能会有较大的偏差（但方差较小）；如果它太"复杂"且参数过多，那么它可能会有较大的方差（但偏差较小）。在上面的例子中，拟合一个二次函数的表现比拟合一阶或五阶多项式的表现要更好。

### 知识准备

接下来，我们开始进入学习理论的探索。除了其本身的趣味性和启发性之外，这一讨论还将帮助我们磨练直觉，并得出关于如何在不同情境下最佳应用学习算法的经验法则。我们还将试图回答几个问题：首先，我们能否正式定义刚刚讨论的偏差/方差权衡？接下来，这将引导我们讨论模型选择方法，例如自动决定拟合训练集的多项式阶数。其次，在机器学习中，我们一般比较关心泛化误差，但大多数学习算法都是拟合它们的模型到训练集上。为什么在训练集上表现良好与否就能告诉我们有关泛化误差的信息？更具体地说，我们能否将训练集上的误差与泛化误差联系起来？最后，是否存在某些条件，在这些条件下我们实际上可以证明学习算法的表现会很好？

#### 切尔诺夫界

我们从两个简单但非常有用的引理开始。

> [!TIP]
> **引理**&ensp;&ensp;**布尔不等式（The union bound）**：设 <span class="math">$ A_1, A_2, \dots, A_k $</span> 是 <span class="math">$ k $</span> 个不同的事件（它们可能不是独立的）。那么
>
> <div class="math">
> $$
> P(A_1 \cup \dots \cup A_k) \leq P(A_1) + \dots + P(A_k)
> $$
> </div>
>
> 在概率论中，并集界通常作为公理提出（因此我们不试图证明它），但它也具有直观意义：<span class="math">$ k $</span> 个事件中任意一个发生的概率最多等于这些事件发生概率之和。
>
> **引理**&ensp;&ensp;**<span class="math">$\small{Hoeffding}$</span> 不等式**：设 <span class="math">$ Z_1, \dots, Z_m $</span> 是从 Bernoulli(<span class="math">$ \phi $</span>) 分布中独立同分布（iid）抽取的随机变量。也就是说，<span class="math">$ P(Z_i = 1) = \phi $</span> 且 <span class="math">$ P(Z_i = 0) = 1 - \phi $</span>。令
>
> <div class="math">
> $$
> \hat{\phi} = \frac{1}{m} \sum_{i=1}^m Z_i
> $$
> </div>
>
> 为这些随机变量的均值，并令任意 <span class="math">$ \gamma > 0 $</span> 为定值。那么
>
> <div class="math">
> $$
> P(|\phi - \hat{\phi}| > \gamma) \leq 2 \exp(-2 \gamma^2 m)
> $$
> </div>
>
> 这个引理（在学习理论中也叫做 **切尔诺夫界（Chernoff bound）** ）说明，如果我们用 <span class="math">$ \hat{\phi} $</span> ——即 <span class="math">$ m $</span> 个 (<span class="math">$ \small{Bernoulli}\ \phi $</span>) 随机变量的平均值——作为 <span class="math">$ \phi $</span> 的估计，那么当 <span class="math">$ m $</span> 足够大时，估计与真实值的 偏差概率会非常小。换句话说，如果你有一枚存在偏差的硬币，其正面朝上的概率是 <span class="math">$ \phi $</span>，那么如果你投掷它 <span class="math">$ m $</span> 次并计算正面朝上的次数的比例，那么该比例将很大概率地接近 <span class="math">$ \phi $</span>（如果 <span class="math">$ m $</span> 足够大）。

仅使用这两个引理，我们就能证明学习理论中一些最深刻和最重要的结果。

为了简易理解，我们将主要将问题证明限制在二分类问题中，其中标签为 <span class="math">$ y \in \\{0, 1\\} $</span>。我们所说的一切将容易推广到其他问题，包括回归和多类别分类问题。

我们假设给定了一个训练集 <span class="math">$ S = \\{(x^{(i)}, y^{(i)}); i = 1, \dots, m\\} $</span>，其中训练样本 <span class="math">$ (x^{(i)}, y^{(i)}) $</span> 是服从某个概率分布 <span class="math">$ \mathcal{D} $</span> 某独立同分布的随机变量。对于一个假设 <span class="math">$ h $</span>，我们定义 <b>训练误差（training error）</b>（在学习理论中也叫做 <b>经验风险（empirical risk）</b> 或 <b>经验误差（empirical error）</b>）为：

<div class="math">
$$
\hat{\varepsilon}(h) = \frac{1}{m} \sum_{i=1}^m 1\{h(x^{(i)}) \neq y^{(i)}\}
$$
</div>

这仅仅是 <span class="math">$ h $</span> 错误分类的训练样本的比例。当我们希望明确指出 <span class="math">$ \hat{\varepsilon}(h) $</span> 对训练集 <span class="math">$ S $</span> 的依赖时，我们可以将其记作 <span class="math">$ \hat{\varepsilon}_S(h) $</span>。同时，定义泛化误差为：

<div class="math">
$$
\varepsilon(h) = P_{(x, y) \sim \mathcal{D}}(h(x) \neq y)
$$
</div>

显然这是一个概率。如果我们现在从分布 <span class="math">$ \mathcal{D} $</span> 中抽取一个新的样本 <span class="math">$ (x, y) $</span>，那么 <span class="math">$ h $</span> 会错误分类它。

需要注意，这里我们有一个预先假设，也就是训练集的数据与要用来检验假设用的数据都服从同一个分布 <span class="math">$ \mathcal{D} $</span>（这是泛化误差定义中的假设）。这有时也被称为 **PAC（Probably Approximately Correct）** 假设之一。

> [!NOTE]
> **PAC** 指的是 <b>"可能近似正确"（probably approximately correct）</b>，这是一个框架和一组假设。基于这些假设，学习理论中的大量结果得以证明。其中最为重要的两个假设就是：**训练集与测试集服从同一分布，以及训练样本的独立性**

#### 经验风险最小化

经验误差 <span class="math">$ \varepsilon(h) $</span> 的这个定义实际上也就相当于，基于分布 <span class="math">$ \mathcal{D} $</span> 给出的一个新的样本 <span class="math">$ (x, y) $</span> 假设模型对该样本分类错误的概率。

考虑线性分类的场景，令 <span class="math">$ h_{\theta}(x) = 1\{\theta^T x \geq 0\} $</span>。一种合理的拟合参数 <span class="math">$ \theta $</span> 的方法是什么？一种方法是尝试最小化训练误差，并选择

<div class="math">
$$
\hat{\theta} = \arg\min_{\theta} \hat{\varepsilon}(h_{\theta})
$$
</div>

我们称这个过程为 **经验风险最小化（Empirical Risk Minimization，ERM**），通过学习算法输出的假设是 <span class="math">$ \hat{h} = h_{\hat{\theta}} $</span>。我们认为 ERM 是最"基本"的学习算法，这也是我们将在这些讲义中主要关注的算法（像逻辑回归这样的算法也可以看作是经验风险最小化的近似形式）。

在我们的学习理论研究中，抽象化假设和具体的参数化，以及是否使用线性分类器等问题是很有帮助的。我们定义用于学习算法的 **假设类(hypothesis class) <span class="math">$ \mathcal{H} $</span>** 是所有该算法考虑的分类器的集合。对于线性分类，<span class="math">$ \mathcal{H} = \{ h_{\theta} : h_{\theta}(x) = 1\{\theta^T x \geq 0\}, \theta \in \mathbb{R}^{n+1} \} $</span>，这是输入特征 <span class="math">$ \mathcal{X} $</span> 上所有分类器的集合，其中决策边界是线性的。更广泛地说，如果我们研究的是神经网络，那么我们可以让 <span class="math">$ \mathcal{H} $</span> 是所有通过某种神经网络架构表示的分类器的集合。

经验风险最小化现在可以被认为是一个在函数类 <span class="math">$ \mathcal{H} $</span> 上的最小化问题，其中由学习算法选择假设：

<div class="math">
$$
\hat{h} = \arg\min_{h \in \mathcal{H}} \hat{\varepsilon}(h)
$$
</div>

### 有限假设类 <span class="math">$ \Large{\mathcal{H}} $</span>

首先考虑一个学习问题，其中我们有一个有限假设类 <span class="math">$ \mathcal{H} = \\{h_1, \dots, h_k\\} $</span>，由 <span class="math">$ k $</span> 个假设组成。因此，<span class="math">$ \mathcal{H} $</span> 就是一个从 <span class="math">$ X $</span> 到 <span class="math">$\\{0, 1\\}$</span> 的 <span class="math">$ k $</span> 个函数集合，经验风险最小化选出的 <span class="math">$ h $</span> 是这 <span class="math">$ k $</span> 个函数中训练误差最小的一个。

我们希望对 <span class="math">$ h $</span> 的泛化误差提供保证。这种策略分为两部分：首先，我们将展示 <span class="math">$ \hat{\varepsilon}(h) $</span> 是所有 <span class="math">$ h $</span> 的可靠估计。其次，我们将证明这意味着 <span class="math">$ \hat{h} $</span> 的泛化误差有上限。

取任意一个固定的 <span class="math">$ h_i \in \mathcal{H} $</span>。考虑一个伯努利随机变量 <span class="math">$ Z $</span>，其分布定义如下。我们将从分布 <span class="math">$ D $</span> 中抽取样本 <span class="math">$ (x, y) $</span>，然后设置 <span class="math">$ Z = 1\\{h_i(x) \neq y\\} $</span>。即我们将抽取一个样本，看 <span class="math">$ Z $</span> 是 <span class="math">$ h_i $</span> 从中产生的误分类数。同样，我们也定义 <span class="math">$ Z_j = 1\\{h_i(x_j) \neq y_j\\} $</span>。由于我们的训练集是从 <span class="math">$ D $</span> 中独立同分布抽取的，<span class="math">$ Z $</span> 和 <span class="math">$ Z_j $</span> 有相同的分布。

我们设定 <span class="math">$ Z $</span> 的期望分类错误概率为 <span class="math">$ \varepsilon(h) $</span>——这正是 <span class="math">$ Z $</span>（以及 <span class="math">$ Z_j $</span>）的期望值。因此，训练误差可以写为：

<div class="math">
$$
\hat{\varepsilon}(h_i) = \frac{1}{m} \sum_{j=1}^m Z_j
$$
</div>

因此，<span class="math">$ \hat{\varepsilon}(h_i) $</span> 正是从伯努利分布（均值为 <span class="math">$ \varepsilon(h_i) $</span>）中抽取的 <span class="math">$ m $</span> 个随机变量 <span class="math">$ Z_j $</span> 的均值。因此，我们可以应用 <span class="math">$\small{Hoeffding}$</span> 不等式，并得到：

<div class="math">
$$
P(|\varepsilon(h_i) - \hat{\varepsilon}(h_i)| > \gamma) \leq 2 \exp(-2\gamma^2 m)
$$
</div>

这表明，对于我们特定的 <span class="math">$ h_i $</span>，训练误差将会接近泛化误差，前提是 <span class="math">$ m $</span> 足够大。但我们不仅想保证 <span class="math">$ \varepsilon(h_i) $</span> 接近 <span class="math">$ \hat{\varepsilon}(h_i) $</span>（高概率），我们希望这对所有 <span class="math">$ h \in \mathcal{H} $</span> 同时成立。为此，设 <span class="math">$ A_i $</span> 表示事件 <span class="math">$|\varepsilon(h_i) - \hat{\varepsilon}(h_i)| > \gamma$</span>。我们已经展示了对于任何特定的 <span class="math">$ A_i $</span>，都有：

<div class="math">
$$
P(A_i) \leq 2 \exp(-2\gamma^2 m)
$$
</div>

因此，使用 **布尔不等式（union bound）**，我们可以得到：

<div class="math">
$$
\begin{aligned}
P(\exists h \in \mathcal{H}.|\varepsilon(h_i) - \hat{\varepsilon}(h_i)| > \gamma) &= P(A_1 \cup \ldots \cup A_k) \\[5pt]
&\leq \sum_{i=1}^k P(A_i) \\[5pt]
&\leq \sum_{i=1}^k 2\exp(-2\gamma^2 m) \\[5pt]
&= 2k\exp(-2\gamma^2 m)
\end{aligned}
$$
</div>

从 1 减去两边，我们得到：

<div class="math">
$$
\begin{aligned}
P(\lnot \exists h \in \mathcal{H}.| \varepsilon(h_i) - \hat{\varepsilon}(h_i)| > \gamma) &= P(\forall h \in \mathcal{H}.|\varepsilon(h_i) - \hat{\varepsilon}(h_i)| \leq \gamma) \\[5pt]
&\geq 1 - 2k\exp(-2\gamma^2 m)
\end{aligned}
$$
</div>

因此，至少以 <span class="math">$1 - 2k\exp(-2\gamma^2 m)$</span> 的概率，<span class="math">$\varepsilon(h)$</span> 将会在 <span class="math">$\hat{\varepsilon}(\hat{h})$</span> 的 <span class="math">$\gamma$</span> 范围内，对所有 <span class="math">$h \in \mathcal{H}$</span> 都成立。这被称为 **一致收敛结果（uniform convergence result）**，因为这个界限同时对 <span class="math">$\mathcal{H}$</span> 中的所有 <span class="math">$h$</span>（而不仅仅是一个）都成立。

正如上面所讨论的，给定 <span class="math">$m$</span> 和 <span class="math">$\gamma$</span> 的特定值，给出了某些 <span class="math">$h \in \mathcal{H}$</span>，<span class="math">$|\varepsilon(h) - \hat{\varepsilon}(h)| > \gamma$</span> 的概率上界。这里有三个关注的量：<span class="math">$m$</span>，<span class="math">$\gamma$</span> 和错误概率；我们可以基于其他两者来限定其中之一。

例如，我们可以问以下问题：给定 <span class="math">$\gamma$</span> 和某个 <span class="math">$\delta > 0$</span>，<span class="math">$m$</span> 需要多大才能保证至少以 <span class="math">$1 - \delta$</span> 的概率，训练误差将在泛化误差的 <span class="math">$\gamma$</span> 范围内？

通过设置 <span class="math">$\delta = 2k\exp(-2\gamma^2 m)$</span> 并求解 <span class="math">$m$</span>，我们发现：

<div class="math">
$$
m \geq \frac{1}{2\gamma^2} \log \frac{2k}{\delta}
$$
</div>

那么，至少以 <span class="math">$1 - \delta$</span> 的概率，我们有 <span class="math">$|\varepsilon(h) - \hat{\varepsilon}(h)| \leq \gamma$</span> 对所有 <span class="math">$h \in \mathcal{H}$</span> 成立。等价地，这表明 <span class="math">$|\varepsilon(h) - \hat{\varepsilon}(h)| > \gamma$</span> 的概率最多是 <span class="math">$\delta$</span>。这个界限告诉我们为了保证一个算法或方法达到一定水平的表现，需要多少训练样本。这也称为算法的 **样本复杂度（sample complexity）**。

上面这个约束条件的关键特性在于，保证这一点所需的训练样本数仅与 <span class="math">$k$</span>，<span class="math">$\mathcal{H}$</span> 中假设的数量对数相关。这个特性稍后会很重要。

同样地，我们也可以固定 <span class="math">$m$</span> 和 <span class="math">$\delta$</span>，求解 <span class="math">$\gamma$</span> 使得之前的等式成立，并展示至少以 <span class="math">$1 - \delta$</span> 的概率对所有 <span class="math">$h \in \mathcal{H}$</span> 有：

<div class="math">
$$
|\varepsilon(h)- \hat{\varepsilon}(h)| \leq \sqrt{\frac{1}{2m} \log \frac{2k}{\delta}}
$$
</div>

现在，让我们假设均一致收敛成立，即对所有 <span class="math">$h \in \mathcal{H}$</span> 有 <span class="math">$|e(h) - \hat{\varepsilon}(h)| \leq \gamma$</span>。我们能证明什么关于我们学习算法的泛化误差，这个算法选择了 <span class="math">$\hat{h} = \arg \min_{h \in \mathcal{H}} \hat{\varepsilon}(h)$</span>？定义 <span class="math">$h^* = \arg\min_{h \in \mathcal{H}} \varepsilon(h)$</span>，这是在 <span class="math">$\mathcal{H}$</span> 中我们可能做到的最好的假设。注意，<span class="math">$h^*$</span> 是我们使用 <span class="math">$\mathcal{H}$</span> 的最佳假设，所以我们能够将 <span class="math">$h^*$</span> 和其他的进行性能比较。我们有：

<div class="math">
$$
\begin{aligned}
\hat{\varepsilon}(\hat{h}) &\leq \hat{\varepsilon}(\hat{h}) + \gamma \\[5pt]
&\leq \hat{\varepsilon}(h^*) + \gamma \\[5pt]
&\leq \varepsilon(h^*) + 2\gamma
\end{aligned}
$$
</div>

上面的第一行用到了定理 <span class="math">$|e(h) - \hat{\varepsilon}(h)| \leq \gamma$</span>（根据我们的一致收敛假设）。第二行用到的定理 <span class="math">$\hat{h}$</span> 是被选择来最小化 <span class="math">$\hat{\varepsilon}(h)$</span>，因此 <span class="math">$\hat{\varepsilon}(h) \leq \hat{\varepsilon}(h^*)$</span>，并且特别地，<span class="math">$\hat{\varepsilon}(h) \leq e(h^*)$</span>。第三行再次使用了一致收敛的假设，以展示 <span class="math">$\hat{\varepsilon}(h^*) \leq e(h^*) + \gamma$</span>。所以，我们展示的是：如果均匀收敛发生，那么 <span class="math">$\hat{h}$</span> 的泛化误差最多比 <span class="math">$\mathcal{H}$</span> 中可能的最佳假设 <span class="math">$h^*$</span> 多 <span class="math">$2\gamma$</span>。

让我们将这一切整理为成一个大的定理：

> [!TIP]
> **定理**&ensp;&ensp;设 **<span class="math">$|\mathcal{H}| = k$</span>，且任意 <span class="math">$m, \delta$</span> 固定。那么至少以 <span class="math">$1 - \delta$</span> 的概率，我们有：**
>
> <div class="math">
> $$
> \hat{\varepsilon}(h) \leq \min_{h \in \mathcal{H}} \varepsilon(h) + 2\sqrt{\frac{1}{2m} \log \frac{2k}{\delta}}
> $$
> </div>

这是通过让 <span class="math">$\gamma$</span> 等于 <span class="math">$\small{\sqrt{}}$</span> 项，使用我们之前的证明，即一致收敛至少以 <span class="math">$1 - \delta$</span> 的概率发生，并指出一致收敛意味着 <span class="math">$\hat{\varepsilon}(h)$</span> 最多比 <span class="math">$e(h^*) = \min_{h \in \mathcal{H}} e(h)$</span> 高 2<span class="math">$\gamma$</span>。

这也量化了我们之前在模型选择中讨论的偏差/方差权衡。特别是，假设我们有一些假设类 <span class="math">$\mathcal{H}$</span>，并正在考虑转向一个更大的假设类 <span class="math">$\mathcal{H}'$</span>。如果我们转向 <span class="math">$\mathcal{H}'$</span>，那么第一个项 <span class="math">$\min_{h \in \mathcal{H}} e(h)$</span> 会改变。

由于如果我们使用一个更大的假设类学习，我们会对一个更大的函数集进行最小化操作，所以误差只能减少（这意味着我们的"偏差"只能减少）。然而，如果 <span class="math">$k$</span> 增加，那么第二个 <span class="math">$\small{\sqrt{}}$</span> 项也会增加。这个增加对应于我们使用更大的假设类时"方差"的增加。

通过固定 <span class="math">$\gamma$</span> 和 <span class="math">$\delta$</span> 并像之前一样求解 <span class="math">$m$</span>，我们也可以得到以下的样本复杂度的约束：

> [!TIP]
> **推论**&ensp;&ensp;**设 <span class="math">$|\mathcal{H}| = k$</span>，且任意 <span class="math">$\gamma, \delta$</span> 固定。那么为使得 <span class="math">$\hat{\varepsilon}(h) \leq \min_{h \in \mathcal{H}} e(h) + 2\gamma$</span> 至少以 <span class="math">$1 - \delta$</span> 的概率成立，只需满足：**
>
> <div class="math">
> $$
> m \geq \frac{1}{2\gamma^2} \log \frac{2k}{\delta} = O\left(\frac{1}{\gamma^2} \log \frac{k}{\delta}\right)
> $$
> </div>

### 无限假设类 <span class="math">$ \Large{\mathcal{\mathcal{H}}} $</span>

我们已经证明了一些有用的定理，适用于有限假设类的情况。但许多假设类，包括由实数参数化的类（如线性分类），实际上包含无限数量的函数。我们能为这种情况证明类似的结果吗？

让我们从某些可能不是"正确"论证的假设开始。存在更好和更通用的论点，但这将有助于磨练我们的直觉。

假设我们有一个 <span class="math">$\mathcal{H}$</span>，它由实数参数化。由于我们使用计算机来表示实数，而IEEE双精度浮点数（C语言中的 double）使用64位来表示一个浮点数，这意味着我们的学习算法，假设我们使用双精度浮点数，是由64位参数化的。因此，我们的假设类实际上最多包含 <span class="math">$2^{64}$</span> 个不同的假设。根据前面的推论，为了 <span class="math">$\hat{\varepsilon}(h^*) + 2\gamma$</span> 至少以 <span class="math">$1 - \delta$</span> 的概率成立，我们有：

<div class="math">
$$
m \geq O\left(\frac{1}{\gamma^2} \log \frac{2^{64}}{\delta}\right) = O\left(\frac{1}{\gamma^2} \log \frac{1}{\delta}\right)
$$
</div>

（<span class="math">$\gamma, \delta$</span> 下标是为了表示最后的大 <span class="math">$O$</span> 隐藏了可能的常数和 <span class="math">$k$</span>）。因此，所需的训练样本数几乎是参数数量的线性倍数。

我们依赖于64位浮点数的事实并不完全令人满意，但结论仍然大致正确：如果我们的目标是最小化训练误差，那么为了学习

当我们使用具有参数的假设类时，通常我们需要在参数数量的线性数量上有训练样本，以便良好地学习。值得注意的是，这些结果是为一个使用经验风险最小化的算法证明的，因此当样本复杂度依赖于 <span class="math">$d$</span>（参数数量）时，这通常适用于大多数试图最小化训练误差或某种近似训练误差的判别学习算法。然而，对于许多非ERM（经验风险最小化）学习算法来说，提供良好的理论保证仍是一个活跃的研究领域。

我们之前论证的另一个稍微不太令人满意的地方是它依赖于 <span class="math">$\mathcal{H}$</span> 的参数化。直观上，这似乎并不合适：我们可以有 <span class="math">$n+1$</span> 个参数 <span class="math">$\theta_0, \dots, \theta_n$</span> 的线性分类器 <span class="math">$\mathcal{H}$</span>，其形式为 <span class="math">$\{h(x) = 1(\theta_0 + \theta_1 x_1 + \dots + \theta_n x_n \geq 0)\}$</span>，但它也可以用 <span class="math">$2+n$</span> 个参数 <span class="math">$u_i, v_i$</span> 表示为 <span class="math">$\{h(x) = 1(u_1^2 - v_1^2 + \dots + (u_n^2 - v_n^2) \geq 0)\}）。尽管这两种形式使用的参数数量不同，但它们定义的都是同一个 <span class="math">$\mathcal{H}$</span>：<span class="math">$n$</span> 维空间中的线性分类器集合。

为了得出更令人满意的论证，让我们定义一些额外的概念。给定一个集合 <span class="math">$S = \{x^{(1)}, \dots, x^{(d)}\}$</span>（与训练集无关的点集），如果对于任何标签集 <span class="math">$\{y^{(1)}, \dots, y^{(d)}\}$</span>，存在某个 <span class="math">$h \in \mathcal{H}$</span>，使得对所有 <span class="math">$i = 1, \dots, d$</span> 都有 <span class="math">$h(x^{(i)}) = y^{(i)}$</span>，我们说 <span class="math">$\mathcal{H}$</span> 能够“打散”<span class="math">$S$</span>。如果 <span class="math">$\mathcal{H}$</span> 可以实现 <span class="math">$S$</span> 上的任意标签化，那么我们说 <span class="math">$\mathcal{H}$</span> 能打散 <span class="math">$S$</span>。

给定一个假设类 <span class="math">$\mathcal{H}$</span>，我们定义它的Vapnik-Chervonenkis维数（简称VC维），为 <span class="math">$\mathcal{H}$</span> 能打散的最大集合的大小。如果 <span class="math">$\mathcal{H}$</span> 能够无限制地打散任意大的集合，那么 <span class="math">$VC(\mathcal{H}) = \infty$</span>。

例如，考虑下面三个点的集合：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/08/28/pAAFb6K.png" data-lightbox="image-7" data-title="3 points">
  <img src="https://s21.ax1x.com/2024/08/28/pAAFb6K.png" alt="3 points" style="width:100%;max-width:150px;cursor:pointer">
 </a>
</div>

这个 <span class="math">$\mathcal{H}$</span>，二维线性分类器的集合 <span class="math">$\{h(x) = 1(\theta_0 + \theta_1 x_1 + \theta_2 x_2 \geq 0)\}$</span>，能否打散上述集合？答案是肯定的。具体来说，我们可以
对于这些点的任何八种可能的标记方式，我们都可以找到一个线性分类器来实现“零训练误差”。此外，可以证明不存在四个点的集合，使得该假设类能够打散它们。因此，这个假设类可以打散的最大集合大小为3，这意味着它的VC维为3。例如，如果我们有三个点在一条直线上（左图），那么找不到一个线性分隔符来对下图中的三个点进行标记（右图）：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/08/28/pAAFOmD.png" data-lightbox="image-7" data-title="linear">
  <img src="https://s21.ax1x.com/2024/08/28/pAAFOmD.png" alt="linear" style="width:100%;max-width:650px;cursor:pointer">
 </a>
</div>
<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/08/28/pAAkC1P.png" data-lightbox="image-7" data-title="VC">
  <img src="https://s21.ax1x.com/2024/08/28/pAAkC1P.png" alt="VC" style="width:100%;max-width:450px;cursor:pointer">
 </a>
</div>

换句话说，根据VC维的定义，为了证明 <span class="math">$VC(\mathcal{H})$</span> 至少是 <span class="math">$d$</span>，我们需要证明存在至少一个大小为 <span class="math">$d$</span> 的集合，<span class="math">$\mathcal{H}$</span> 可以打散它。

以下定理由Vapnik提出，这也许是学习理论中最重要的定理之一：

定理。给定 <span class="math">$\mathcal{H}$</span>，并设 <span class="math">$d = VC(\mathcal{H})$</span>。那么至少以 <span class="math">$1 - \delta$</span> 的概率，对于所有 <span class="math">$h \in \mathcal{H}$</span>，我们有：

<div class="math">
$$
|e(h) - \hat{\varepsilon}(h)| \leq O\left(\sqrt{\frac{d}{m} \log \frac{m}{d} + \frac{1}{m} \log \frac{1}{\delta}}\right)
$$
</div>

因此，至少以 <span class="math">$1 - \delta$</span> 的概率，我们也有：

<div class="math">
$$
\hat{\varepsilon}(h) \leq e(h^*) + O\left(\sqrt{\frac{d}{m} \log \frac{m}{d} + \frac{1}{m} \log \frac{1}{\delta}}\right)
$$
</div>

换句话说，如果一个假设类具有有限的VC维度，那么随着 <span class="math">$m$</span> 变大，均匀收敛会发生，这允许我们用 <span class="math">$e(h^*)$</span> 来给出 <span class="math">$\hat{\varepsilon}(h)$</span> 的上界。我们也有以下推论：

推论。对于 <span class="math">$|e(h) - \hat{\varepsilon}(h)| \leq \gamma$</span> 而言，至少以 <span class="math">$1 - \delta$</span> 的概率成立，只需要 <span class="math">$m = O(\frac{d}{\gamma^2})$</span>。

总体来说，使用 <span class="math">$\mathcal{H}$</span> 是线性的，在VC维度假设下，大多数假设类的参数数量（假设“合理”的参数化）也大致是参数数量的线性倍数。将这些因素结合起来，我们得出结论，通常所需的训练样本数与 <span class="math">$\mathcal{H}$</span> 的参数数量大致成线性关系。