# 第七章&ensp;学习理论

在先前的章节里，我们基本已经对机器学习中的学习方法有了基本的了解。接下来，我们将以各种方法为引，对机器学习中的核心理论进行分析，以强化我们的理论学习，帮助我们进行误差和性能分析。

通过下面的学习，应该重点掌握：

* 偏差/方差权衡
* 切尔诺夫界
* VC 维

- - -

### 偏差/方差权衡

在讨论线性回归时，我们讨论了是否应该拟合“简单”模型（例如线性模型 <span class="math"> $ y = \theta_0 + \theta_1 x $</span>），还是更“复杂”的模型（例如多项式 <span class="math">$ y = \theta_0 + \theta_1 x + \dots + \theta_5 x^5 $</span>）。我们看到以下示例：

（此处为三张图像，分别展示了不同拟合复杂度的曲线）

拟合5次多项式到数据上（最右边的图像）并没有得到一个好的模型。具体来说，即使5次多项式对训练集中的 <span class="math">$ y $</span>（比如房价）和 <span class="math">$ x $</span>（比如居住面积）做出了非常好的预测，我们并不期望这个模型能很好地预测训练集中没有出现的房屋价格。换句话说，从训练集中学习到的内容并没有很好地 **泛化（generalize）** 到其他房屋。**泛化误差（generalization error）**（稍后将正式定义）是指假设在不一定在训练集中的例子上产生的预期误差。

最左边的模型和最右边的模型都存在较大的泛化误差。然而，这两个模型的问题来源却非常不同。如果 <span class="math">$ y $</span> 和 <span class="math">$ x $</span> 之间的关系不是线性的，即使我们将线性模型拟合到非常大量的训练数据上，线性模型仍然无法准确捕捉数据中的结构。非正式地，我们将 **偏差（bias）** 定义为模型的预期泛化误差，即使我们将其拟合到非常（假设是无限大）大的训练集上也是如此。因此，对于上述问题，线性模型存在较大的偏差，并且可能出现欠拟合（即，未能捕捉到数据中的结构）。

除了偏差之外，泛化误差的第二个组成部分是模型拟合过程中的 **方差（variance）**。具体来说，当我们像右边图中那样拟合五次多项式时，存在较大风险，即我们拟合了训练集中偶然存在的模式，而这些模式并没有反映 $ x $ 和 $ y $ 之间的更广泛关系。这可能是因为在训练集中，我们碰巧得到了稍微比平均价格更贵的房子，或者稍微比平均价格更便宜的房子，等等。通过拟合这些“虚假”的训练集模式，我们可能会得到一个具有大泛化误差的模型。在这种情况下，我们说模型具有较大的方差。

因此，在偏差和方差之间存在权衡。如果我们的模型太"简单"并且参数很少，那么它可能会有较大的偏差（但方差较小）；如果它太“复杂”且参数过多，那么它可能会有较大的方差（但偏差较小）。在上面的例子中，拟合一个二次函数的表现比拟合一阶或五阶多项式的表现都要好。

### 知识准备

在这一部分的讲义中，我们开始进入学习理论的探索。除了其本身的趣味性和启发性之外，这一讨论还将帮助我们磨练直觉，并得出关于如何在不同情境下最佳应用学习算法的经验法则。我们还将试图回答几个问题：首先，我们能否正式定义刚刚讨论的偏差/方差权衡？接下来，这将引导我们讨论模型选择方法，例如自动决定拟合训练集的多项式阶数。其次，在机器学习中
泛化误差是我们关心的，但大多数学习算法都是拟合它们的模型到训练集上。为什么在训练集上表现良好能告诉我们有关泛化误差的信息？更具体地说，我们能否将训练集上的误差与泛化误差联系起来？最后，是否存在某些条件，在这些条件下我们实际上可以证明学习算法的表现会很好？

我们从两个简单但非常有用的引理开始。

**引理**（并集界）：设 <span class="math">$ A_1, A_2, \dots, A_k $</span> 是 <span class="math">$ k $</span> 个不同的事件（它们可能不是独立的）。那么

<div class="math">
$$
P(A_1 \cup \dots \cup A_k) \leq P(A_1) + \dots + P(A_k)。
$$
</div>

在概率论中，并集界通常作为公理提出（因此我们不试图证明它），但它也具有直观意义：<span class="math">$ k $</span> 个事件中任意一个发生的概率最多等于这些事件发生概率之和。

**引理**（Hoeffding 不等式）：设 <span class="math">$ Z_1, \dots, Z_m $</span> 是从 Bernoulli(<span class="math">$ \phi $</span>) 分布中独立同分布（iid）抽取的随机变量。也就是说，<span class="math">$ P(Z_i = 1) = \phi $</span> 且 <span class="math">$ P(Z_i = 0) = 1 - \phi $</span>。令

<div class="math">
$$
\hat{\phi} = \frac{1}{m} \sum_{i=1}^m Z_i
$$
</div>

为这些随机变量的均值，并令任意 <span class="math">$ \gamma > 0 $</span> 为定值。那么

<div class="math">
$$
P(|\phi - \hat{\phi}| > \gamma) \leq 2 \exp(-2 \gamma^2 m)。
$$
</div>

这个引理（在学习理论中也叫做**切尔诺夫界（Chernoff bound）**）说明，如果我们用 <span class="math">$ \hat{\phi} $</span> ——即 <span class="math">$ m $</span> 个 Bernoulli(<span class="math">$ \phi $</span>) 随机变量的平均值——作为 <span class="math">$ \phi $</span> 的估计，那么当 <span class="math">$ m $</span> 足够大时，估计与真实值的偏差概率会非常小。换句话说，如果你有一枚失衡硬币，其正面朝上的概率是 <span class="math">$ \phi $</span>，那么如果你投掷它 <span class="math">$ m $</span> 次并计算正面朝上的次数的比例，那么该比例将很大概率地接近 <span class="math">$ \phi $</span>（如果 <span class="math">$ m $</span> 足够大）。

仅使用这两个引理，我们就能证明学习理论中一些最深刻和最重要的结果。

为了简化我们的叙述，我们将注意力限制在二元分类问题中，其中标签为 <span class="math">$ y \in \{0, 1\} $</span>。我们所说的一切将容易推广到其他问题，包括回归和多类别分类问题。

我们假设给定了一个训练集 <span class="math">$ S = \{(x^{(i)}, y^{(i)}); i = 1, \dots, m\} $</span>，其中训练样本 <span class="math">$ (x^{(i)}, y^{(i)}) $</span> 是从某个概率分布 <span class="math">$ \mathcal{D} $</span> 独立同分布地抽取的。对于一个假设 <span class="math">$ h $</span>，我们定义 **训练误差(training error)**（在学习理论中也叫做 **经验风险(empirical risk)** 或 **经验误差(empirical error)**）为：

<div class="math">
$$
\hat{\epsilon}(h) = \frac{1}{m} \sum_{i=1}^m 1\{h(x^{(i)}) \neq y^{(i)}\}。
$$
</div>

这仅仅是 <span class="math">$ h $</span> 错误分类的训练样本的比例。当我们希望明确指出 <span class="math">$ \hat{\epsilon}(h) $</span> 对训练集 <span class="math">$ S $</span> 的依赖时，我们可以将其记作 <span class="math">$ \hat{\epsilon}_S(h) $</span>。我们还定义了泛化误差为：

<div class="math">
$$
\epsilon(h) = P_{(x, y) \sim \mathcal{D}}(h(x) \neq y)。
$$
</div>

也就是说，这是一个概率：如果我们现在从分布 <span class="math">$ \mathcal{D} $</span> 中抽取一个新的样本 <span class="math">$ (x, y) $</span>，那么 <span class="math">$ h $</span> 会错误分类它。

需要注意的是，我们假设训练数据是从与我们用于评估假设的相同的分布 <span class="math">$ \mathcal{D} $</span> 中抽取的（这是泛化误差定义中的假设）。这有时也被称为 **PAC（Probably Approximately Correct）** 假设之一。

考虑线性分类的场景，令 <span class="math">$ h_{\theta}(x) = 1\{\theta^T x \geq 0\} $</span>。一种合理的拟合参数 <span class="math">$ \theta $</span> 的方法是什么？一种方法是尝试最小化训练误差，并选择

<div class="math">
$$
\hat{\theta} = \arg\min_{\theta} \hat{\epsilon}(h_{\theta})。
$$
</div>

我们称这个过程为 **经验风险最小化（Empirical Risk Minimization，ERM**），通过学习算法输出的假设是 <span class="math">$ \hat{h} = h_{\hat{\theta}} $</span>。我们认为 ERM 是最“基本”的学习算法，这也是我们将在这些讲义中关注的算法（像逻辑回归这样的算法也可以看作是经验风险最小化的近似形式）。

在我们的学习理论研究中，抽象化假设和具体的参数化，以及是否使用线性分类器等问题是有帮助的。我们定义用于学习算法的 **假设类(hypothesis class) <span class="math">$ \mathcal{H} $</span>** 是所有该算法考虑的分类器的集合。对于线性分类，<span class="math">$ \mathcal{H} = \{ h_{\theta} : h_{\theta}(x) = 1\{\theta^T x \geq 0\}, \theta \in \mathbb{R}^{n+1} \} $</span>，这是输入域 <span class="math">$ \mathcal{X} $</span> 上所有分类器的集合，其中决策边界是线性的。更广泛地说，如果我们研究的是神经网络，那么我们可以让 <span class="math">$ \mathcal{H} $</span> 是所有通过某种神经网络架构表示的分类器的集合。

经验风险最小化现在可以被认为是一个在函数类 <span class="math">$ \mathcal{H} $</span> 上的最小化问题，其中学习算法选择假设：

<div class="math">
$$
\hat{h} = \arg\min_{h \in \mathcal{H}} \hat{\epsilon}(h)。
$$
</div>

**PAC** 指的是“可能近似正确”（probably approximately correct），这是一个框架和一组假设，基于这些假设，学习理论中的大量结果得以证明。其中，训练集和测试集来自同一分布的假设，以及独立抽取的训练样本的假设是最为重要的。