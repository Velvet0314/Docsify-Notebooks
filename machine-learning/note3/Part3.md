# 第三章&ensp;二分类

在学习完回归问题后，接下来我们将对应地了解分类问题。对于一个分类问题，最简单的情况就是**二分类（binary classification）**。于是，我们将从二分类问题开始，逐步过渡到多分类问题。

通过下面的学习，应该对以下知识有着基本的了解：

* 感知机
* 牛顿法

通过下面的学习，应该重点掌握：

* 逻辑回归

- - -

### ⭐逻辑回归

对于一个分类问题，预测值的多样性就变得无意义。于是，我们将预测值规定在 <span class="math">$ y \in \\{0,1\\} $</span> 上。相应地，我们也需要变更我们的假设函数：

<div class="math">
$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$
</div>

<div class="math">
$  \quad   \quad   \quad   \quad   \quad  \text{where} $
</div>

<div class="math">
$$
g(z) = \frac{1}{1 + e^{-z}}
$$
</div>

其中，<span class="math">$ g(z) $</span> 被称为 **logistic 函数** 或 **sigmoid 函数**。其图像如图：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/05/26/pklcnER.png" data-lightbox="image-3" data-title="sigmoid function">
  <img src="https://s21.ax1x.com/2024/05/26/pklcnER.png" alt="igmoid function" style="width:100%;max-width:450px;cursor:pointer">
 </a>
</div>

关于假设函数的选择有诸多原因，这里不展开叙述。但是其中之一值得我们注意，那就是 sigmoid 函数的求导特性：

<div class="math">
$$
\begin{aligned}
g'(z) &= \frac{d}{dz} \frac{1}{1 + e^{-z}} \\[5pt]
&= \frac{1}{(1 + e^{-z})^2} (e^{-z}) \\[5pt]
&= \frac{1}{(1 + e^{-z})} \cdot \left(1 - \frac{1}{(1 + e^{-z})}\right) \\[5pt]
&= g(z)(1 - g(z))
\end{aligned}
$$
</div>

接下来，我们仍然通过在线性回归中使用过方法来构建模型——极大似然估计。

同样地，我们需要一些前提假设以易于后续的模型构建：

<div class="math">
$$
\begin{aligned}
&P(y = 1 \mid x; \theta) = h_\theta(x) \\[5pt]
&P(y = 0 \mid x; \theta) = 1 - h_\theta(x)
\end{aligned}
$$
</div>

通过整理，改写成下面这个形式：

<div class="math">
$$
p(y \mid x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}  \quad   \quad  y \in \{0,1\}
$$
</div>

假设 $ m $ 个训练样本是独立的，于是我们写出似然函数：

<div class="math">
$$
\begin{aligned}
L(\theta) &= p(\vec{y} \mid X; \theta) \\[5pt]
&= \prod_{i=1}^m p(y^{(i)} \mid x^{(i)}; \theta) \\[5pt]
&= \prod_{i=1}^m (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1-y^{(i)}}
\end{aligned}
$$
</div>

同样地，使用对数似然函数来简化计算：

<div class="math">
$$
\begin{aligned}
\ell(\theta) &= \log L(\theta) \\[5pt]
&= \sum_{i=1}^m y^{(i)} \log h(x^{(i)}) + (1 - y^{(i)}) \log (1 - h(x^{(i)}))
\end{aligned}
$$
</div>

如何最大化 <span class="math">$ L(\theta) $</span> 呢？还是与之前一样，采用 **梯度上升（gradient ascent）**，即更新策略为：<span class="math">$ \theta := \theta + \alpha \nabla_\theta \ell(\theta)\ $</span>。

> [!NOTE]
> 注意这里使用的是 **梯度上升**，因为我们这里是是在“**最大化**”一个函数，而非在梯度下降中“**最小化**”代价函数。

与 SGD 相似地，对于其中一个训练样本：

<div class="math">
$$
\begin{aligned}
\quad \quad \quad \quad \quad \quad \quad \quad \frac{\partial}{\partial \theta_j} \ell(\theta) &\color{orange}{= \left( \frac{y}{g(\theta^T x)} - (1 - y) \frac{1}{1 - g(\theta^T x)} \right) \frac{\partial}{\partial \theta_j} g(\theta^T x)} \\
&  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \color{red}{\rightarrow \text{求导的链式法则：} \frac{\partial}{\partial \theta_j} \ell(\theta) = \frac{\partial \ell(\theta)}{\partial h_\theta(x)} \cdot \frac{\partial h_\theta(x)}{\partial \theta_j}} \\
&=  \left( \frac{y}{g(\theta^T x)} - (1 - y) \frac{1}{1 - g(\theta^T x)} \right) \color{lightgreen}{g(\theta^T x) (1 - g(\theta^T x)) x_j} \\
&  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \color{red}{\rightarrow \text{求导的链式法则：} \frac{\partial h_\theta(x)}{\partial \theta_j} = \frac{\partial g(\theta^T x)}{\partial (\theta^T x)} \cdot \frac{\partial (\theta^T x)}{\partial \theta_j}} \\
&= (y (1 - g(\theta^T x)) - (1 - y) g(\theta^T x)) x_j \\[10pt]
&= (y - h_\theta(x)) x_j
\end{aligned}
$$
</div>

于是，我们得到了如下的更新规则：
<div class="math">
$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}
$$
</div>

这样，我们就得到了二分类的学习方法：**逻辑回归（logistic regression）**。如果我们将其与 LMS 更新规则进行比较，我们会发现它看起来是相同的。但这并不是相同的算法，因为 $<span class="math"> h_\theta(x^{(i)}) $</span> 现在定义为 <span class="math">$ \theta^T x^{(i)} $</span> 的非线性函数。尽管如此，有点令人惊讶的是，我们最终为一个完全不同的算法和学习问题得到了相同的更新规则。这是巧合，还是背后有更深层的原因？后续的 GLM 模型会解开我们的困惑。

#### 总结：逻辑回归

在逻辑回归模型中，分类取决于概率。我们可以把这个概率称为决策分数，指 <span class="math">$ \theta^T x $</span> 的值。该值经过 sigmoid 函数转换为概率。具体来说：

- 当 <span class="math">$ \theta^T x = 0 $</span> 时，sigmoid 函数输出为 0.5，即分类边界，表示分类的不确定性最大。
- 当 <span class="math">$ \theta^T x > 0 $</span> 时，sigmoid 函数输出大于 0.5，模型倾向于预测正类（标签 1）。
- 当 <span class="math">$ \theta^T x < 0 $</span> 时，sigmoid 函数输出小于 0.5，模型倾向于预测负类（标签 0）。

### 感知机

这里我们简单地介绍另一种二分类模型：**感知机（perception）**。

感知机是一种线性二分类模型，其决策边界由一个线性函数来定义：

<div class="math">
$$
g(z) = \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$
</div>

使用同样的更新规则：

<div class="math">
$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}
$$
</div>

感知机是一个较为早期的模型，其相对于逻辑回归也没有严格意义上的概率证明。

### 牛顿法

除开梯度上升，我们还可以考虑另一种最大化 $ \ell(\theta) $ 的方法：**牛顿法（Newton's method）**。

牛顿法是一种用于求解非线性方程的迭代算法，也被广泛应用于优化问题中寻找函数的极值点。其基本思想是利用函数的导数信息，通过线性近似逐步逼近函数的根或极值点。

给出一个初始点 <span class="math">$ x_0 $</span>，利用切线去逼近函数的根，其迭代公式为：

<div class="math">
$$
x_{k+1} := x_k - \frac{f(x_k)}{f'(x_k)}
$$
</div>

重复迭代，直到 $ \left| x_{k+1}-x_k \right| $ 或 $ \left| f(x_{k+1}) \right| $ 小于给定上限 $ \varepsilon $

下图是牛顿法的一个示例：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/05/27/pk19UXT.png" data-lightbox="image-2" data-title="Newton's method">
  <img src="https://s21.ax1x.com/2024/05/27/pk19UXT.png" alt="Newton's method" style="width:100%;max-width:1000px;cursor:pointer">
 </a>
</div>

为了最大化 <span class="math">$ \ell(\theta) $</span>，显然我们需要求解 <font size=4><span class="math">$ \frac{\partial \ell(\theta)}{\partial \theta} = 0 $</span></font>，也就是求 <span class="math">$ \ell(\theta) $</span> 的驻点 <span class="math">$ \ell'(\theta) = 0 $</span>。

所以相应地，更新规则为：

<div class="math">
$$
\theta := \theta - \frac{\ell'(\theta)}{\ell''(\theta)}
$$
</div>

由于我们在逻辑回归中采用的 <span class="math">$ \theta $</span> 是向量形式，于是我们将牛顿法推广到高维，通常也称为 **牛顿-拉弗森法（Newton-Raphson Method）**。此时，更新规则变更为：

<div class="math">
$$
\theta := \theta - H^{-1}\nabla_\theta\ell(\theta)
$$
</div>

这里的 $ H $ 称为 **海森矩阵（Hessian matrix）**，其定义为：

<div class="math">
$$
H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}
$$
</div>

牛顿法具有二次收敛速度，所以需要的迭代次数相比于梯度下降会显著减少；但是，牛顿法的开销较大，同时对于初始点的选取也是十分敏感，更容易出现收敛到局部极值或发散的情况。