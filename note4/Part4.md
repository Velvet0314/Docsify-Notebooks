# 第四章&ensp;广义线性模型

通过前面的学习，我们对于分类问题有了一个大致的了解。但是之前我们一直都是进行着最简单的二分类，从现在开始，我们将要考虑进行更多类别的划分。在此之前，我们将会对之前的疑问进行解答，同时引入一个全新的概念：**广义线性模型（Generalized Linear Models）**，并通过它来构建我们的多分类模型。

通过下面的学习，应该重点掌握：

* 指数族
* GLM 的构建原理
* 从 GLM 的角度分析线性回归和逻辑回归
* Softmax 回归

- - -

### 指数族

为了理解 GLM，首先我们需要了解 **指数族（exponential family）**。

指数族是概率分布的集合，它由概率分布的指数次及其参数组成。

<div class="math">
$$
p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta))
$$
</div>

以下是参数解释：

- $ \eta \ \text{自然参数（natural parameter）} $：这是分布的参数，控制分布的形状。

- $ T(y) \ \text{充分统计量（sufficient statistic）} $：它是关于数据 $ y $ 的函数。对于我们常见的许多分布，$ T(y) $ 就是 $ y $ 本身。

- $ a(\eta) \ \text{对数分割函数（log partition function）} $：这个函数用于确保概率分布的归一化，即分布 $ p(y; \eta) $ 对所有可能的 $ y $ 求和或积分等于 1。

- $ b(y) $：这是关于 $ y $ 的函数，通常用于调整分布的形状。

当 $ T,a,b $ 都固定时，即定义了一个以 $ \eta $ 为参数的函数分布族。当我们改变 $ \eta $ 时，我们得到该族内的不同分布。

我们所熟悉的 **伯努利分布（Bernoulli distribution）** 与 **高斯分布（Gaussian distribution）** 都属于指数族。

接下来，我们将从数学角度来验证上述两个分布是属于指数族的。但在这之前，我们仍简要介绍一下伯努利分布。

具有一个均值 $ \phi $ 的伯努利分布，记作 $ Bernoulli(\phi) $，定义了 $ y \in \\{0,1\\} $ 的概率分布,使得：

<div class="math">
$$
\begin{aligned}
&p(y = 1; \phi) = \phi \\[5pt]
&p(y = 0; \phi) = 1 - \phi
\end{aligned}
$$
</div>

整理一下，也就是：

<div class="math">
$$
p(y;\phi) = \phi^y(1-\phi)^{1-y} \ \ \ \ \ \ \ \ y \in \{0,1\}
$$
</div>

当我们改变 $ \phi $ 时，我们得到具有不同均值的伯努利分布。


其中，$ \phi $ 表示 $ y=1 $ 的概率。

下面，我们将伯努利分布写成如下形式：

<div class="math">
$$
\begin{aligned}
p(y; \phi) &= \phi^y (1 - \phi)^{1 - y} \\[5pt]
&= \exp(y \log \phi + (1 - y) \log (1 - \phi)) \\[5pt]
&= \exp \left( \left( \log \left( \frac{\phi}{1 - \phi} \right) \right) y + \log (1 - \phi) \right)
\end{aligned}
$$
</div>

因此，自然参数由 $ \eta = \log(\phi / (1 - \phi)) $ 给出。有趣的是，如果我们通过 $ \eta $ 求解 $ \phi $ 来逆转这个定义，我们得到 $ \phi = 1 / (1 + e^{-\eta}) $。这是我们所熟悉的 Sigmoid 函数！这将在我们将逻辑回归作为 GLM 推导时再次出现。为了完成将伯努利分布作为指数族分布的公式化，我们还有：

<div class="math">
$$
\begin{aligned}
&T(y) = y \\[5pt]
&a(\eta) = -\log(1 - \phi) = \log(1 + e^{\eta}) \\[5pt]
&b(y) = 1 \\[5pt]
\end{aligned}
$$
</div>

