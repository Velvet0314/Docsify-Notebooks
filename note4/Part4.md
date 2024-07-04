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
p(y; \phi) &= \color{orange}{\phi^y (1 - \phi)^{1 - y}} \\[5pt]
& \quad  \quad  \quad  \quad  \quad \color{red}{\rightarrow \text{对数变换：}\phi^y = e^{log\phi^y} = e^{ylog\phi}} \\[5pt]
&= \exp(y \log \phi + (1 - y) \log (1 - \phi)) \\[5pt]
&= \exp \left( \left( \log \left( \frac{\phi}{1 - \phi} \right) \right) y + \log (1 - \phi) \right)
\end{aligned}
$$
</div>

因此，自然参数由 $ \eta = \log(\phi / (1 - \phi)) $ 给出。有趣的是，如果我们通过 $ \eta $ 求解 $ \phi $ 来逆转这个定义，我们将得到 $ \phi = 1 / (1 + e^{-\eta}) $。这是我们所熟悉的 Sigmoid 函数！这将在我们将逻辑回归作为 GLM 推导时再次出现。为了完成将伯努利分布作为指数族分布的公式化，我们还有：

<div class="math">
$$
\begin{aligned}
&T(y) = y \\[5pt]
&a(\eta) = -\log(1 - \phi) = \log(1 + e^{\eta}) \\[5pt]
&b(y) = 1
\end{aligned}
$$
</div>

至此，伯努利分布就被改写成了指数族形式。下面，我们将把高斯分布也写成指数族的形式。

在对线性回归进行推导时，$ \sigma^2 $ 的值对最后的结果是没有影响的，所以为了简化推导过程，我们一般取 $ \sigma^2 = 1 $。

<div class="math">
$$
\begin{aligned}
p(y; \mu) &= \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{1}{2}(y - \mu)^2 \right) \\[5pt]
&= \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{1}{2}y^2 \right) \cdot \exp \left( \mu y - \frac{1}{2} \mu^2 \right)
\end{aligned}
$$
</div>

然后，改写成指数族的形式：

<div class="math">
$$
\begin{aligned}
\eta &= \mu \\[4pt]
T(y) &= y \\[4pt]
a(\eta) &= \frac{\mu^2}{2} \\[4pt]
&= \frac{\eta^2}{2} \\[4pt]
b(y) &= \left( \frac{1}{\sqrt{2\pi}} \right) \exp \left( -\frac{y^2}{2} \right)
\end{aligned}
$$
</div>

其实，许多我们熟悉的分布都是属于指数族的，例如：

> [!NOTE]
> * **多项式分布（multinomial）：用来对多元分类问题进行建模**
> * **泊松分布（Poisson）：用来对计数过程进行建模，如网站的访客数量、商店的顾客数量等**
> * **伽马分布（gamma）和指数分布（exponential）：用来对时间间隔进行建模，如等车时间等**
> * **β 分布（beta）和 Dirichlet 分布（Dirichlet）：用于概率分布**
> * **Wishart 分布（Wishart）：用于协方差矩阵分布**

### GLM 的构建

一般地，考虑一个分类或回归问题，我们希望预测某个随机变量 $ y $ 的值，它是 $ x $ 的函数。为了为这个问题推导出一个 GLM 模型，我们将对给定 $ x $ 的 $ y $ 的条件分布以及我们的模型做以下三个假设：

> [!TIP]
>**1. $ y \mid x; \theta \sim \text{ExponentialFamily}(\eta) $。即，给定 $ x $ 和 $\theta$，$ y $ 服从某个参数为 $\eta$ 的指数族分布。**
>
>**2. 给定 $ x $，我们的目标是预测 $ T(y) $ 的期望值。一般地，我们有 $ T(y) = y $，也就是 $ h(x) = \hat{y} = \mathbb{E}[y \mid x] $。**
>
> **注意，逻辑回归和线性回归中 $ h_\theta(x) $ 均满足这个条件。例如，在对数回归中，我们有 $ h_\theta(x) = p(y = 1 \mid x; \theta) = 0 \cdot p(y = 0 \mid x; \theta) + 1 \cdot p(y = 1 \mid x; \theta) = \mathbb{E}[y \mid x; \theta] $）。**
>
>**3. 自然参数 $\eta$ 和输入 $ x $ 线性相关：$\eta = \theta^T x $。（如果 $\eta$ 是向量值，那么 $\eta_i = \theta_i^T x $）。**

其中，第三个假设可能看起来是最不合理的，但它可以被认为是我们设计 GLM 的一个“设计选择”，而不是严格意义上的假设。这三个假设会使我们能够得到一个优秀的学习算法，即 GLM。此外，结果模型通常对不同类型的分布非常有效。例如，GLM 的构建可以在线性回归与逻辑回归中都学习到不错的模型。

#### 最小二乘法

为了证明最小二乘法是广义线性模型（GLM）的一个特例，考虑目标变量 $ y $，在 GLM 术语中也称为 **响应变量（response variable**）是连续的情形，我们将 $ y $ 的条件分布建模为 $ x $ 为高斯分布 $ \mathcal{N}(\mu, \sigma^2) $（这里，$\mu$ 可能依赖于 $ x $）。因此，我们让上述的指数族分布 $ \text{ExponentialFamily}(\eta) $ 是高斯分布。如我们之前所见，在高斯分布作为指数族分布的形式中，我们有 $\mu = \eta$。所以，我们有

<div class="math">
$$
\begin{aligned}
h_\theta(x) &= \mathbb{E}[y \mid x; \theta] \\[4pt]
&= \mu \\[4pt]
&= \eta \\[4pt]
&= \theta^T x
\end{aligned}
$$
</div>

第一个等式来自假设 2；第二个等式来自 $ y \mid x; \theta \sim \mathcal{N}(\mu, \sigma^2) $ ，因此其期望值由 $\mu$ 给出；第三个等式来自假设 1（以及我们之前推导出的在高斯作为指数族分布的形式中 $\mu = \eta$ 的结果）；最后一个等式来自假设 3。

#### 逻辑回归

现在我们开始考虑逻辑回归。这里我们依然以二分类为例，所以 $ y \in \\{0, 1\\} $。由于 $ y $ 是二值的，因此选择伯努利分布来建模给定 $ x $ 的 $ y $ 的条件分布。在我们将伯努利分布作为指数族分布的形式中，我们有 $ \phi = 1/(1 + e^{-\eta}) $。此外，如果 $ y \mid x; \theta \sim \text{Bernoulli}(\phi) $，那么 $ \mathbb{E}[y \mid x; \theta] = \phi $。因此，按照与最小二乘法相似的推导，我们得到：

<div class="math">
$$
\begin{aligned}
h_\theta(x) &= \mathbb{E}[y \mid x; \theta] \\[4pt]
&= \phi \\[4pt]
&= \frac{1}{1 + e^{-\eta}} \\[4pt]
&= \frac{1}{1 + e^{-\theta^T x}}
\end{aligned}
$$
</div>

因此，这给出了形如 $ h_\theta(x) = \displaystyle{\frac{1}{1 + e^{-\theta^T x}}} $ 的假设函数。如果你之前想知道我们是如何得到逻辑回归的函数形式的，那么这个假设就给出了答案。一旦我们假设 $ y $ 条件于 $ x $ 是伯努利分布，它就作为 GLM 和指数族分布自然导出的结果。

进一步地，以 $ \eta $ 为参数的函数 $ g $ （$ g(\eta) = \mathbb{E}[T(y) ; \eta] $）叫做 **正则响应函数（canonical response function）**。作为 $ T(y) $ 期望值的反函数 $ g^{-1} $，叫做 **正则关联函数（canonical link function）**。不同的指数族的族内分布的正则响应函数是不同的。例如，高斯分布的正则响应函数就是 **判别函数（identify function）**，伯努利分布的正则响应函数就是 **逻辑函数（logistic function）**。

#### ⭐Softmax 回归

##### 类比推广

考虑一个分类问题，其中响应变量 $ y $ 可以取 $ k $ 个值中的任意一个，即 $ y \in \{1, 2, \ldots, k\} $。响应变量仍然是离散的，但现在可以取超过两个值。我们因此将其建模为多项分布。

由之前的二分类，我们可以将伯努利分布转化为多项式分布。

下面，利用 GLM 对多项数据建模进行推导。为此，我们首先将多项式表示为指数族分布。

为了对 $ k $ 个可能结果进行多项式参数化，可以使用 $ k $ 个参数 $\phi_1, \ldots, \phi_k$ 来指定每个结果的概率。然而，这些参数存在着冗余，或者标准地说，它们 **不是独立的**。

> [!WARNING]
> **因为已知任意 $ k-1 $ 个 $\phi$ 的值,就可以唯一确定最后一个值，因为它们必须满足 $ \sum_{i=1}^{k} \phi_i = 1 $**

因此，我们将仅使用 $ k-1 $ 个参数 $\phi_1, \ldots, \phi_{k-1}$ 来对多项式进行参数拟合，其中 $ \phi_i = p(y = i; \phi) $，同时 $ p(y = k; \phi) = 1 - \sum_{i=1}^{k-1} \phi_i $。我们还将设 $\phi_k = 1 - \sum_{i=1}^{k-1} \phi_i$，但我们应记住这不是一个参数，并且它完全由 $\phi_1, \ldots, \phi_{k-1}$ 定义的。

为了将多项式表示为一个指数族分布，我们将 $ T(y) \in \mathbb{R}^{k-1} $ 定义如下：

<div class="math">
$$
\begin{aligned}
T(1) = \begin{bmatrix}
1 \\
0 \\
0 \\
\vdots \\
0
\end{bmatrix}, \quad T(2) = \begin{bmatrix}
0 \\
1 \\
0 \\
\vdots \\
0
\end{bmatrix}, \quad T(3) = \begin{bmatrix}
0 \\
0 \\
1 \\
\vdots \\
0
\end{bmatrix}, \quad \ldots, \quad T(k-1) = \begin{bmatrix}
0 \\
0 \\
0 \\
\vdots \\
1
\end{bmatrix}, \quad T(k) = \begin{bmatrix}
0 \\
0 \\
0 \\
\vdots \\
0
\end{bmatrix}
\end{aligned}
$$
</div>

与之前的模型不同，我们不将 $ T(y) $ 定义为 $ y $ 的值：$ T(y) $ 现在是一个 $ k-1 $ 维向量，而不再是实数。我们将记 $ (T(y))_i $ 表示向量 $ T(y) $ 的第 $ i $ 个元素。

这里我们引入一个非常有用的记号：**指示函数（indicator function）** $ 1\\{\cdot\\} $

在其参数为真时值为 1，在其参数为假时值为 0，即 $ 1 \\{True\\} = 1, 1 \\{False\\} = 0 $。例如，$ 1\\{2 = 3\\} = 0 $， $ 1\\{3 = 5 - 2\\} = 1 $。

于是，我们可以将 $ T(y) $ 与 $ y $ 的关系写为 $ (T(y))_i = 1\\{y = i\\} $。$ (T(y))_i $ 的其期望为 $ \mathbb{E}[T(y)_i] = P(y = i) = \phi_i $。

##### 验证并构建 GLM

下面开始推导多项式分布的指数族形式：

<div class="math">
$$
\begin{aligned}
\quad \quad \quad \quad \quad \quad p(y; \phi) &= \phi_1^{1\{y=1\}} \phi_2^{1\{y=2\}} \cdots \phi_k^{1\{y=k\}} \\[5pt]
&= \phi_1^{1\{y=1\}} \phi_2^{1\{y=2\}} \cdots \phi_k^{1 - \sum_{i=1}^{k-1} 1\{y=i\}} \\[5pt]
&= \color{orange}{\phi_1^{T(y)_1} \phi_2^{T(y)_2} \cdots \phi_k^{1 - \sum_{i=1}^{k-1} T(y)_i}} \\[5pt]
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \color{red}{\rightarrow \text{转化为独立的变量}} \\[3pt]
&= \exp \left( (T(y))_1 \log (\phi_1) + (T(y))_2 \log (\phi_2) + \cdots + \left( 1 - \sum_{i=1}^{k-1} (T(y))_i \right) \log (\phi_k) \right) \\[5pt]
&= \exp \left( (T(y))_1 \log (\phi_1 / \phi_k) + (T(y))_2 \log (\phi_2 / \phi_k) + \cdots + (T(y))_{k-1} \log (\phi_{k-1} / \phi_k) + \log (\phi_k) \right) \\[5pt]
&= b(y) \exp \left( \eta^T T(y) - a(\eta) \right)
\end{aligned}
$$
</div>

其中：

<div class="math">
$$
\begin{aligned}
\eta &= \begin{bmatrix} \log (\phi_1 / \phi_k) \\ \log (\phi_2 / \phi_k) \\ \vdots \\ \log (\phi_{k-1} / \phi_k) \end{bmatrix} \\[5pt]
a(\eta) &= -\log (\phi_k) \\[5pt]
b(y) &= 1
\end{aligned}
$$
</div>

至此，我们验证了该多项式分布属于指数族。

关联函数（$ \text{for}\ i = 1, \ldots, k $）如下：

<div class="math">
$$
\eta_i = \log \frac{\phi_i}{\phi_k}
$$
</div>

为了方便起见，我们还定义了 $ \eta_k = \log (\phi_k / \phi_k) = 0 $。为了求反关联函数以得到响应函数，有：

<div class="math">
$$
\begin{align}
e^{\eta_i} &= \frac{\phi_i}{\phi_k} \\[5pt]
\phi_k e^{\eta_i} &= \phi_i \tag{1} \\[5pt]
\phi_k \sum_{i=1}^{k} e^{\eta_i} &= \sum_{i=1}^{k} \phi_i = 1
\end{align}
$$
</div>

其中， $ \phi_k = \displaystyle{\frac{1}{\sum_{i=1}^{k} e^{\eta_i}}} $，这可以代入回方程（1）得到响应函数：

<div class="math">
$$
\begin{align}
\phi_i = \frac{e^{\eta_i}}{\sum_{j=1}^{k} e^{\eta_j}}
\end{align}
$$
</div>

这个将 $ \eta $ 映射到 $ \phi $ 的函数被称为 **softmax函数**。

为了构建模型，我们使用之前提到的假设 3，即 $ \eta_i $ 与 $ x $  线性相关。因此，令 $ \eta_i = \theta_i^T x $（$ \text{for}\ i = 1, \ldots, k-1 $）。

其中 $ \theta_1, \ldots, \theta_{k-1} \in \mathbb{R}^{n+1} $ 是模型的参数。

为了易于记录，可定义 $ \theta_k = 0 $，这样就有 $ \eta_k = \theta_k^T x = 0 $。因此，我们的模型假设在给定 $ x $ 的条件下 $ y $ 的条件分布为：

<div class="math">
$$
\begin{align}
p(y = i | x; \theta)&= \phi_i \\[5pt]
&= \frac{e^{\eta_i}}{\sum_{j=1}^k e^{\eta_j}} \\[5pt]
&= \frac{e^{\theta_i^T x}}{\sum_{j=1}^k e^{\theta_j^T x}} \tag{2}
\end{align}
$$
</div>

这个模型适用于 $ y \in \\{1, \ldots, k\\} $ 的分类问题，称为 **softmax 回归**，是用于二分类的逻辑回归的推广。

我们的假设函数将输出：

<div class="math">
$$
\begin{aligned}
h_\theta(x) &= \mathbb{E}[T(y) | x; \theta] \\[5pt]
&= \mathbb{E} \left[ \begin{array}{c}
1 \{ y = 1 \} \\[5pt]
1 \{ y = 2 \} \\[5pt]
\vdots \\[5pt]
1 \{ y = k - 1 \}
\end{array} \middle| x; \theta \right] \\[5pt]
&= \left[ \begin{array}{c}
\phi_1 \\[5pt]
\phi_2 \\[5pt]
\vdots \\[5pt]
\phi_{k-1}
\end{array} \right] \\[5pt]
&= \left[ \begin{array}{c}
\frac{\exp(\theta_1^T x)}{\sum_{j=1}^k \exp(\theta_j^T x)} \\[5pt]
\frac{\exp(\theta_2^T x)}{\sum_{j=1}^k \exp(\theta_j^T x)} \\[5pt]
\vdots \\[5pt]
\frac{\exp(\theta_{k-1}^T x)}{\sum_{j=1}^k \exp(\theta_j^T x)}
\end{array} \right]
\end{aligned}
$$
</div>

换句话说，我们的假设将输出对于 $ i = 1, \ldots, k $ 的每一个值的估计概率 $ p(y = i | x; \theta) $。

> [!WARNING]
> **即使上面定义的 $ h_\theta(x) $ 仅为 $ k - 1 $ 维，但是显然可以得到 $ p(y = k | x; \theta) $ 为 $ 1 - \sum_{i=1}^{k-1} \phi_i $。**

##### 参数估计

最后，我们进行参数估计。类似于对普通最小二乘法和逻辑回归的推导，如果有一个包含 $ m $ 个样本的训练集 $ \\{(x^{(i)}, y^{(i)}); i = 1, \ldots, m\\} $ 并且希望学习这个模型的参数 $ \theta_i $。

首先写出其对数似然函数：

<div class="math">
$$
\begin{aligned}
\ell(\theta) &= \sum_{i=1}^{m} \log p(y^{(i)} | x^{(i)}; \theta) \\[5pt]
&= \sum_{i=1}^{m} \log \prod_{l=1}^{k} \left( \frac{e^{\theta_l^T x^{(i)}}}{\sum_{j=1}^{k} e^{\theta_j^T x^{(i)}}} \right)^{1\{y^{(i)} = l\}}
\end{aligned}
$$
</div>

为了得到上述公式的第二行，采用了方程（2）中给出的 $ p(y | x; \theta) $ 的定义。

我们现在可以通过最大化 $ \ell(\theta )$ 来获得参数 $ \theta $ 的最大似然估计。使用的方法可以是我们之前学习过的梯度上升或牛顿法。