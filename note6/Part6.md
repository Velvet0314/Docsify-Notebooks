# 第六章&ensp;支持向量机

在这一章节中我们将进行学习 **支持向量机（support vector machine）**。支持向量机一度被许多人认为是最好的 “即用式” 监督学习算法。通俗来讲，SVM 是一种二分类模型。接下来，我们将深入 SVM，来看看其是否是如此的优秀。

通过下面的学习，应该对以下知识有着基本的了解：

* 拉格朗日对偶
* 正则化

通过下面的学习，应该重点掌握：

* SVM 的定义
* 函数边界与几何边界
* 最优边界决策器
* 核 Kernel
* SMO 算法

- - -

### 边界：直观感受

让我们回到最开始的二分类问题。当时我们采用的解决方式是逻辑回归。

考虑逻辑回归，其中概率 <span class="math">$ p(y = 1|x; \theta) $</span> 由 <span class="math">$ h_\theta(x) = g(\theta^T x) $</span> 建模。

如果 <span class="math">$ h_\theta(x) \geq 0.5 $</span>，或等效地，如果 <span class="math">$ \theta^T x \geq 0 $</span>，我们就会在输入 <span class="math">$ x $</span> 上预测 1。考虑一个正样本（<span class="math">$ y = 1 $</span>），<span class="math">$ \theta^T x $</span> 越大，<span class="math">$ h_\theta(x) = p(y = 1|x; w, b) $</span> 也越大，因此我们对标签是 1 的 “置信度” 也越高。

我们可以认为如果 <span class="math">$ \theta^T x \gg 0 $</span>，我们的预测非常确定 <span class="math">$ y = 1 $</span>。类似地，我们认为逻辑回归在 <span class="math">$ \theta^T x \ll 0 $</span> 时对 <span class="math">$ y = 0 $</span> 做出高置信度的预测。考虑到训练集，我们似乎找到了一个很好的拟合训练数据的模型：

**如果我们能找到使 <span class="math">$ \theta^T x^{(i)} \gg 0 $</span> 的 <span class="math">$ \theta $</span> 确信 <span class="math">$ y^{(i)} = 1 $</span>，且 <span class="math">$ \theta^T x^{(i)} \ll 0 $</span> 确信 <span class="math">$ y^{(i)} = 0 $</span>。**

为了更直观地感受 **边界（margin）**，样例如下图所示。

其中 <span class="math">$ × $</span> 表示正训练样本，<span class="math">$ \text{o} $</span> 表示负训练样本，决策边界是由方程 <span class="math">$ \theta^T x = 0 $</span> 给出的直线，也称为 <strong>分隔超平面（separating hyperplane）</strong>

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/07/08/pkfPHRP.png" data-lightbox="image-6" data-title="margin">
  <img src="https://s21.ax1x.com/2024/07/08/pkfPHRP.png" alt="margin" style="width:100%;max-width:375px;cursor:pointer">
 </a>
</div>

请注意，点 A 离决策边界很远。如果我们被要求在 A 处对 <span class="math">$ y $</span> 的值进行预测，似乎我们应该确信 <span class="math">$ y = 1 $</span>。相反，点 C 非常靠近决策边界，虽然它位于我们会预测 <span class="math">$ y = 1 $</span> 的决策边界的一侧，但似乎只需对决策边界稍作改动，就很容易使我们的预测变为 <span class="math">$ y = 0 $</span>。因此，我们对 A 的预测比对 C 的预测置信度更高，而点 B 位于这两种情况之间。

进一步地，如果一个点离分隔超平面很远，那么其可能在我们的预测中更置信。也就是说，训练样本需要离边界尽可能地远。这样会使得我们的模型更加有效。

### SVM 定义

为了易于后续学习 SVM，我们首先需要引入一些新的符号来讨论分类问题。

我们将考虑一个用于二元分类问题的线性分类器，其标签为 $ y $ 并且特征为 $ x $。从现在开始，我们将使用 <span class="math">$ y \in \\{-1, 1\\} $</span> 来表示类标签。此外，我们不再用向量 <span class="math">$ \theta $</span> 来描述我们的线性分类器，而是使用参数 <span class="math">$ w, b $</span>。于是将分类器写为：

<div class="math">
$$
h_{w,b}(x) = g(w^T x + b)
$$
</div>

如果 <span class="math">$ z \geq 0 $</span>，那么有 <span class="math">$ g(z) = 1 $</span>；否则 <span class="math">$ g(z) = -1 $</span>。

这种 <span class="math">$ w, b $</span> 符号让我们可以更明确地单独处理截距项 <span class="math">$ b $</span> 和其他参数。同时，也放弃了之前让 <span class="math">$ x_0 = 1 $</span> 作为输入特征向量中的一个额外项。在这里，<span class="math">$ b $</span> 取代了 <span class="math">$ \theta_0 $</span>，而 <span class="math">$ w $</span> 取代了 <span class="math">$[ \theta_1 \ldots \theta_n ]^T$</span>。

> [!WARNING]
> **根据我们上面定义的 <span class="math">$ g $</span>，我们的分类器将直接预测 1 或 −1（参考感知机算法），而不是首先估计 <span class="math">$ y = 1 $</span> 的概率（这是逻辑回归所做的）。**

> [!TIP]
> 逻辑回归与支持向量机的定义区别<br>
> <table>
> <thead>
> <tr>
> <th>特性</th>
> <th>逻辑回归</th>
> <th>支持向量机</th>
> </tr>
> </thead>
> <tbody>
> <tr>
> <td><strong>标签和输出</strong></td>
> <td>输出标签为 {0, 1}，输出为概率</td>
> <td>输出标签为 {-1, 1}，直接预测类别</td>
> </tr>
> <tr>
> <td><strong>决策函数</strong></td>
> <td>使用 sigmoid 函数预测概率，连续</td>
> <td>使用符号函数（感知机），离散</td>
> </tr>
> </tbody>
> </table>

### 边界

#### 函数边界

对于一个给定的训练样本 <span class="math">$ \\{x^{(i)},y^{(i)}\\} $</span>，定义关于 <span class="math">$ (w,b) $</span> 关于样本的 **函数边界（functional margin）**为：

<div class="math">
$$
\hat{\gamma}^{(i)} = y^{(i)}(w^T x + b)
$$
</div>

对于支持向量机，其决策分数通常表现为 <span class="math">$ w^T x + b $</span>：
* 当 <span class="math">$ w^T x + b = 0 $</span> 时，点恰好在决策边界上，即函数边界。
* 当 <span class="math">$ w^T x + b > 0 $</span> 时，模型预测样本属于正类（<span class="math">$ y = 1 $</span>）。
* 当 <span class="math">$ w^T x + b < 0 $</span> 时，模型预测样本属于负类（<span class="math">$ y = -1 $</span>）。

通过之前的学习，我们得出结论：要使结果置信度更高，那么就要求 <span class="math">$ \left| w^T x + b \right| $</span> 需要尽可能的大。

考虑两种情况，当 <span class="math">$ y = 1 $</span> 时，有 <span class="math">$ w^T x + b \gg 0 $</span>；当 <span class="math">$ y = -1 $</span> 时，有 <span class="math">$ w^T x + b \ll 0 $</span>。

但是针对于函数边界，存在一个特殊的问题。

如果我们用 <span class="math">$ 2w $</span> 和 <span class="math">$ 2b $</span> 替换 <span class="math">$w$</span> 和 <span class="math">$b$</span> （进行放缩），就会得到 <span class="math">$ g(w^T x + b) = g(2w^T x + 2b) $</span>，这不会改变 <span class="math">$ h_{w,b}(x) $</span> 预测结果的正负。然而，用 <span class="math">$(2w, 2b)$</span> 替换 <span class="math">$(w, b)$</span> 却会使函数边界增加 2 倍。也就是说，通过任意的对于 <span class="math">$w$</span> 和 <span class="math">$b$</span> 的放缩，使得函数边界在衡量置信度上失去了意义（但实际的决策平面未改变）。为了解决这个问题，我们希望能够添加某种规范化条件，比如 <span class="math">$\|\|w\|\|_2 = 1$</span>。也就是说，我们可能会用 <span class="math">$(\displaystyle{\frac{w}{\|\|w\|\|_2}}, \displaystyle{\frac{b}{\|\|w\|\|_2}})$</span> 替换 <span class="math">$(w, b)$</span>，并相应地考虑 <span class="math">$(\displaystyle{\frac{w}{\|\|w\|\|_2}}, \displaystyle{\frac{b}{\|\|w\|\|_2}})$</span> 的函数边界。

> [!NOTE]
> **分隔平面 <span class="math">$ w^T x + b = 0 $</span> 描述的是一个通过原点的超平面，其方向由向量 <span class="math">$ w $</span> 确定，偏移由 <span class="math">$ b $</span> 确定。当 <span class="math">$ w $</span> 和 <span class="math">$ b $</span> 同时乘以同一个非零常数时，超平面的方向和偏移比例保持不变，因此超平面本身也保持不变。**

给定训练集 <span class="math">$ S = \\{(x^{(i)}, y^{(i)}) ; i = 1, \dots, m\\} $</span>，我们还定义了 <span class="math">$ (w, b) $</span> 相对于 <span class="math">$ S $</span> 的函数边界为最小的那个个体训练样本的函数边界。通过 <span class="math">$ \hat{\gamma} $</span> 表示，写为：

<div class="math">
$$
\hat{\gamma} = \min_{i=1,...,m} \hat{\gamma}^{(i)}
$$
</div>

#### 几何边界

首先从下面这张图开始：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/07/09/pkf3wU1.png" data-lightbox="image-6" data-title="geometric margin">
  <img src="https://s21.ax1x.com/2024/07/09/pkf3wU1.png" alt="geometric margin" style="width:100%;max-width:425px;cursor:pointer">
 </a>
</div>

决策边界对应于 <span class="math">$ (w, b) $</span> 如图所示，同时还标记了向量 <span class="math">$ \vec{w} $。注意 <span class="math">$ \vec{w} $</span> 与超平面是正交的。也就是说，<span class="math">$ \vec{w} $</span> 表示了样本到超平面的一个法向量。

考虑点 <span class="math">$ A $</span>，它代表某个训练样本的输入 <span class="math">$ x^{(0)} $</span> 与标签 <span class="math">$ y^{(0)} = 1 $</span>。其到决策边界的距离 <span class="math">$ \gamma^{(0)} $</span> 由线段 </span>$AB$</span> 给出。

如何确定 <span class="math">$\gamma^{(0)} $</span> 的值呢？已知 <span class="math">$ \displaystyle{\frac{\vec{w}}{\|\vec{w}\|}} $</span> 是一个指向 <span class="math">$ \vec{w} $</span> 同一方向的单位向量。由点 <span class="math">$ A $</span> 表示 <span class="math">$ x^{(0)} $</span>，得点 <span class="math">$ B $</span> 为 <span class="math">$ x^{(0)} - \gamma^{(0)} \cdot (\displaystyle{\frac{\vec{w}}{\|\vec{w}\|}}) $</span>。

> [!NOTE]
> 由于点 <span class="math">$ B $</span> 在决策边界上，从 <span class="math">$ A $</span> 到 <span class="math">$ B $</span> 的向量（<span class="math">$ \vec{AB} $）必须与 <span class="math">$ \vec{w} $</span> 方向一致，并且是垂直于边界的。因此，<span class="math">$ \vec{AB} $</span> 可以表达为 <span class="math">$ \vec{w} $</span> 单位向量的某个标量倍数，即：
>
> <div class="math">
> $$
> \vec{AB} = -\gamma^{(0)} \cdot \left(\frac{\vec{w}}{\|\vec{w}\|}\right)
> $$
> </div>
>
> 这里使用负号，因为我们从 <span class="math">$ A $</span> 沿 <span class="math">$ \vec{w} $</span> 的方向“移动”到边界，若 <span class="math">$ \vec{w} $</span> 从 <span class="math">$ A $</span> 指向超平面外，则移动方向为 <span class="math">$ \vec{w} $</span> 的反方向。
>点 <span class="math">$ B $</span> 的位置 <span class="math">$ x^{(B)} $</span> 由下式给出：
>
> <div class="math">
> $$
> x^{(B)} = x^{(0)} + \vec{AB} = x^{(0)} - \gamma^{(0)} \cdot \left(\frac{\vec{w}}{\|\vec{w}\|}\right)
> $$
> </div>

考虑一般情况，决策边界上的所有点都满足方程 <span class="math">$ w^T x + b = 0 $</span>。因此，有：

<div class="math">
$$
w^T \left(x^{(i)} - \gamma^{(i)} \cdot \frac{w}{\|w\|}\right) + b = 0
$$
</div>

求解 <span class="math">$ \gamma^{(i)} $</span> 得：

<div class="math">
$$
\gamma^{(i)} = \frac{w^T x^{(i)} + b}{\|w\|} = \left(\frac{w}{\|w\|}\right)^T x^{(i)} + \frac{b}{\|w\|}
$$
</div>

这是针对 <span class="math">$ A $</span> 在图中所处 "正样本" 位置的情况计算得出。更普遍地，我们定义一个训练样本 <span class="math">$ \\{x^{(i)},y^{(i)}\\} $</span> 相对于 <span class="math">$ (w, b) $</span> 的 **几何边界（geometric margin）** 为：

<div class="math">
$$
\gamma^{(i)} = y^{(i)} \left( \left(\frac{w}{\|w\|}\right)^T x^{(i)} + \frac{b}{\|w\|} \right)
$$
</div>

注意，如果 <span class="math">$ \|w\| = 1 $</span>，那么函数边界就等于几何边界。同时，几何边界具有放缩不变性。如果我们用 <span class="math">$ 2w $</span> 和 <span class="math">$ 2b $</span> 替换 <span class="math">$ w $</span> 和 <span class="math">$ b $</span>，几何边界不会改变。因为经过前面的推导，几何边界实际上是样本点到超平面的欧几里得距离。

几何边界所具有放缩不变性会为 SVM 带来许多优势：

* **模型简化与正则化**：
调整权重向量 <span class="math">$ w $</span> 和偏置 <span class="math">$ b $</span> 的尺度不会影响模型的决策边界。这允许在模型训练过程中引入正则化约束，如限制 <span class="math">$ \|w\| $</span> 的大小，从而防止过拟合。
* **参数选择的灵活性**：
在部署模型到不同硬件或软件环境时，可能需要对模型参数进行缩放以满足特定的性能或资源使用要求，缩放不变性确保了这种调整不会影响模型的表现。

最后，给定一个训练集 <span class="math">$ S = \\{(x^{(0)},y^{(0)}); i = 1, \ldots, m\\} $</span>，我们也定义了 <span class="math">$ (w, b) $</span> 相对于 <span class="math">$ S $</span> 的几何边界为个体训练样本几何边界的最小值：

<div class="math">
$$
\gamma = \min_{i=1,\ldots,m} \gamma^{(i)}
$$
</div>

> [!TIP]
> **<font size=4>函数边界与几何边界的联系</font>**<br>
> **函数边界：**
> <div class="math">
> $$
> \hat{\gamma}^{(i)} = y^{(i)}(w^T x + b)
> $$
> </div>
>
> **几何边界：**
> <div class="math">
> $$
> \gamma^{(i)} = \frac{y^{(i)}(w^T x + b)}{|| w ||}
> $$
> </div>
>
> **当 <span class="math">$ || w || = 1 $</span> 时，函数边界与几何边界相等。**

### 拉格朗日对偶

#### 数学准备

##### 函数的凹凸性

###### 凸函数

一个定义在实数上的函数 <span class="math">$ f(x) $</span> 被称为 **凸函数（convex function）**，如果对于所有 <span class="math">$ x_1, x_2 $</span> 在其定义域内，并且对于所有 <span class="math">$ \lambda $</span> 满足 <span class="math">$ 0 \leq \lambda \leq 1 $</span>，都有：

<div class="math">
$$
f(\lambda x_1 + (1-\lambda) x_2) \leq \lambda f(x_1) + (1-\lambda) f(x_2)
$$
</div>

这个性质被称为函数的 **凸性（convex）** 。这意味着函数的任意两点之间的线段都位于函数图形的上方或与之相接。一个典型的凸函数例子是 <span class="math">$ f(x) = x^2 $</span>。

###### 凹函数

与凸函数相对的是 **凹函数（concave function）**，定义几乎相同，只是不等式的方向相反。一个函数 <span class="math">$ f(x) $</span> 是凹函数，如果对于所有 <span class="math">$ x_1, x_2 $</span> 在其定义域内，并且对于所有 <span class="math">$ \lambda $</span> 满足 <span class="math">$ 0 \leq \lambda \leq 1 $</span>，都有：

<div class="math">
$$
f(\lambda x_1 + (1-\lambda) x_2) \geq \lambda f(x_1) + (1-\lambda) f(x_2)
$$
</div>

这表示函数的任意两点之间的线段都位于函数图形的下方或与之相接。一个常见的凹函数例子是 <span class="math">$ f(x) = \log(x) $</span>，在其定义域 <span class="math">$ x > 0 $</span> 上。

> [!NOTE]
> <font size=4>**函数凹凸性的判断**</font>
>
> * **几何直观**：凸函数类似于一个向上开口的碗，而凹函数则像一个向下开口的碗。
> * **导数判定**：
>   * **凸函数**：如果一个函数的二阶导数（如果存在的话）是非负的（<span class="math">$ f''(x) \geq 0 $</span>），那么这个函数是凸函数。
>   * **凹函数**：如果一个函数的二阶导数是非正的（<span class="math">$ f''(x) \leq 0 $</span>），那么这个函数是凹函数。

##### 无约束的优化问题

优化问题顾名思义，也就是要找寻一个值来最小化或者最大化一个目标函数，最简单的形式如下：

<div class="math">
$$
\min_{x \in \mathbb{R}^n} f(x)
$$
</div>

这个式子表达的含义是以 <span class="math">$x$</span> 为自变量，找到 <span class="math">$f(x)$</span> 的最小值。如果想要求的最大值，可以把 <span class="math">$ f $</span> 进行一个转换，这样把 <span class="math">$\text{max}_x f(x)$</span> 等价转换为 <span class="math">$\text{min}_x -f(x)$</span> 。

注意，<span class="math">$x$</span> 并非一个标量数值，而是 <span class="math">$n$</span> 维实数空间上的一个向量。

在线性回归中，我们需要最小化代价函数。对于这样的问题，我们通常是求导寻找驻点，即找到 <span class="math">$x$</span> 使得 <span class="math">$\nabla f(x') = 0$</span>。

当然我们也知道，只有当 <span class="math">$f$</span> 是凸函数的时候，这样才能得到全局最优解，否则会出现局部最优解的情况。下面的讨论都是假设函数为凸函数。

##### 带有等式约束的优化问题

一般地，我们进行优化函数总会带有一些限制条件，称为约束。带有等式约束的优化问题是一种更为复杂的优化任务，其中除了要最小化或最大化一个目标函数外，还需要满足一个或多个等式约束条件。

给定一个目标函数 <span class="math">$ f(x) $</span>，其中 <span class="math">$ x \in \mathbb{R}^n $</span>，以及一组等式约束 <span class="math">$ g_i(x) = 0 $</span>，其中 <span class="math">$ i = 1, \ldots, m $</span>，问题可以定义为：

<div class="math">
$$
\begin{aligned}
\qquad \qquad &\min_{x \in \mathbb{R}^n} f(x) \\[5pt]
&\text{s.t.} \quad g_i(x) = 0, \quad i = 1, \ldots, m
\end{aligned}
$$
</div>

##### 拉格朗日乘数法

在面对具有约束的优化问题时，我们常常使用 **拉格朗日乘数法（Lagrange Multiplier Method）** 来解决。

拉格朗日乘数法的基本思想是通过引入 **拉格朗日乘子（Lagrange multiplier）**，将带有约束条件的优化问题转化为一个不带约束的问题。

假设有一个需要优化的目标函数 <span class="math">$ f(x) $</span>，同时还有一个或多个约束条件 <span class="math">$ g_i(x) = 0 $</span>。定义拉格朗日函数以结合约束：
<div class="math">
$$
L(x, \lambda) = f(x) + \sum_{i=1}^{m} \lambda_i g_i(x)
$$
</div>

其中，<span class="math">$ \lambda_i $</span> 是对应于第 <span class="math">$ i $</span> 个约束条件 <span class="math">$ g_i(x) $</span> 的拉格朗日乘子。

为了找到目标函数 <span class="math">$ f(x) $</span> 在约束条件 <span class="math">$ g_i(x) = 0 $</span> 下的极值点，我们需要求解拉格朗日函数 <span class="math">$ L(x, \lambda) $</span> 对变量 <span class="math">$ x $</span> 和乘子 <span class="math">$ \lambda $</span> 的偏导数：
<div class="math">
$$
\left\{
\begin{aligned}
& \frac{\partial L}{\partial x_j} = 0 \\[5pt]
& \frac{\partial L}{\partial \lambda_i} = 0
\end{aligned}
\right.
$$
</div>

这些方程被称为拉格朗日方程。

通过求解方程组，可以得到 <span class="math">$ x $</span> 和 <span class="math">$ \lambda $</span> 的值，这些值给出了目标函数在满足约束条件下的极值点。

> [!TIP]
> <font size=4>**拉格朗日乘数法中的参数解释**</font><br>
> **拉格朗日乘子** <span class="math">$\lambda $</span>：它是用来引入约束条件 <span class="math">$ g_i(x) = 0 $</span> 到目标函数 <span class="math">$ f(x) $</span> 中的系数。每个约束条件 <span class="math">$ g_i(x) $</span> 对应一个拉格朗日乘子 <span class="math">$ \lambda_i $</span>，通常用来确定约束条件对目标函数的影响。<br>
> **驻点**：通过拉格朗日函数的偏导数为零的点（驻点），可以找到目标函数在约束条件下的可能极值点。这些驻点满足了同时优化目标函数 <span class="math">$ f(x) $</span> 和约束条件 <span class="math">$ g_i(x) = 0 $</span> 的要求。

##### 仿射函数

一个函数 <span class="math">$ f: \mathbb{R}^n \to \mathbb{R}^m $</span> 被称为 **仿射函数（affine function）**，如果它可以表示为：

<div class="math">
$$
f(x) = A x + b
$$
</div>

其中：
* <span class="math">$ A $</span> 是一个 <span class="math">$ m \times n $</span> 的常数矩阵，
* <span class="math">$ b $</span> 是一个 <span class="math">$ m $</span> 维常向量，
* <span class="math">$ x $</span> 是一个 <span class="math">$ n $</span> 维向量。

例如：

一个从 <span class="math">$ \mathbb{R}^n $</span> 到 <span class="math">$ \mathbb{R}^1 $</span> 的典型仿射函数可以写作：

<div class="math">
$$
f(x) = a^T x + b
$$
</div>

其中：
* <span class="math">$ x $</span> 是一个 <span class="math">$ n $</span> 维向量，即 <span class="math">$ x = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^n $</span>。
* <span class="math">$ a = (a_1, a_2, \ldots, a_n) $</span> 是一个 <span class="math">$ n $</span> 维常向量，即 <span class="math">$ a \in \mathbb{R}^n $</span>。
* <span class="math">$ b $</span> 是一个常数，即 <span class="math">$ b \in \mathbb{R} $</span>。

仿射函数具有如下一个重要的特性。对于任意 <span class="math">$ x, y \in \mathbb{R}^n $</span> 和任意 <span class="math">$ \lambda \in [0, 1] $</span>：

<div class="math">
$$
f(\lambda x + (1-\lambda) y) = a^T(\lambda x + (1-\lambda) y) + b = \lambda a^T x + (1-\lambda) a^T y + b
$$
</div>

这可以被重写为：

<div class="math">
$$
f(\lambda x + (1-\lambda) y) = \lambda f(x) + (1-\lambda) f(y)
$$
</div>

这表明仿射函数是同时满足凸函数和凹函数的条件。

##### 凸优化

凸优化问题是指在凸集上求解凸函数的优化问题。一般形式的凸优化问题可以表示为：

<div class="math">
$$
\begin{aligned}
&\min_{x \in \mathbb{R}^n} f(x) \\[5pt]
&\text{subject to} \quad g_i(x) \leq 0, \quad i = 1, 2, \ldots, m \\[5pt]
&\quad \quad \quad \quad \quad h_j(x) = 0, \quad j = 1, 2, \ldots, p
\end{aligned}
$$
</div>

其中：
* <span class="math">$ f(x) $</span> 是凸函数，称为目标函数。
* <span class="math">$ g_i(x) $</span> 是凸函数，表示不等式约束。
* <span class="math">$ h_j(x) $</span> 是仿射函数，表示等式约束。
* <span class="math">$ \mathbb{R}^n $</span> 中的 <span class="math">$ x $</span> 是优化变量。

凸优化问题具有一个重要的性质：**任何凸优化问题都具有全局最优解**。

##### 对偶问题

###### 线性规划

在进一步了解对偶问题前，我们需要先了解 **线性规划（linear programming）**。

一个典型的线性规划问题可以表示为：

* **目标函数**：需要进行最大化或最小化的线性函数。

<div class="math">
$$
\text{maximize} \quad c^T x \quad \text{or} \quad \text{minimize} \quad c^T x
$$
</div>

其中 <span class="math">$ c $</span> 和 <span class="math">$ x $</span> 均为向量。

* **约束条件**：这些是问题的限制条件，形式为线性等式或不等式。

<div class="math">
$$
Ax \leq b, \quad Ax = b, \quad \text{or} \quad Ax \geq b
$$
</div>

其中 <span class="math">$ A $</span> 是一个矩阵，<span class="math">$ b $</span> 是一个向量。

* **变量的非负限制**：在大多数线性规划问题中，决策变量需要非负。

<div class="math">
$$
x \geq 0
$$
</div>

###### 问题定义

在线性规划早期发展中最重要的发现是对偶问题，即每一个线性规划问题（称为原始问题 P）有一个与它对应的对偶线性规划问题（称为对偶问题 D）。

在线性规划问题中一个经典的问题的描述如下：

材工厂有两种原料 A, B，而且能用其生产两种产品：

* 生产第一种产品需要2个A和4个B，能获利6；
* 生产第二种产品需要3个A和2个B，能获利4；
* 一共有100个A和120个B，问这工厂的最多获利？用数学表达式描述如下：

<div class="math">
$$
\begin{align*}
\text{maximize} \quad & 6x_1 + 4x_2 \\[5pt]
\text{subject to} \quad & 2x_1 + 3x_2 \leq 100 \\[5pt]
& 4x_1 + 2x_2 \leq 120
\end{align*}
$$
</div>

工厂除了拿原料生产成产品卖掉这条出路外，还有一种方法是直接将原料卖掉。但是这种情况只存在于把原料卖掉赚的钱比生产成产品赚的钱多，才去会这样做。那么最低可以接受多少的价格呢？假设原料A和B的单价分别为：<span class="math">$w_1$</span> 和 <span class="math">$w_2$</span>，那么可以用数学表达式描述如下：

<div class="math">
$$
\begin{align*}
\text{minimize} \quad & 100w_1 + 120w_2 \\[5pt]
\text{subject to} \quad & 2w_1 + 4w_2 \geq 6 \\[5pt]
& 3w_1 + 2w_2 \geq 4
\end{align*}
$$
</div>

这两个问题互为对偶问题，分别称之为原问题（P）和对偶问题（D）。

每一个线性规划问题都存在一个与其对偶的问题，**在求出一个问题解的同时，也给出了另一个问题的解**。当对偶问题比原始问题有较少约束时，求解对偶规划比求解原始规划要方便得多。

#### 定义

对于拉格朗日对偶问题，我们如下处理。考虑下面的优化问题，我们称之为原问题：

<div class="math">
$$
\begin{align*}
\text{minimize} \quad & f(w) \\[5pt]
\text{subject to} \quad & g_i(w) \leq 0, \ i = 1, \dots, k \\[5pt]
& h_j(w) = 0, \ j = 1, \dots, l
\end{align*}
$$
</div>

为了解决它，我们首先定义了广义的拉格朗日函数：

<div class="math">
$$
\mathcal{L}(w, \alpha, \beta) = f(w) + \sum_{i=1}^{k} \alpha_i g_i(w) + \sum_{i=1}^{l} \beta_i h_i(w)
$$
</div>

这里，<span class="math">$\alpha_i$</span> 和 <span class="math">$\beta_i$</span> 是拉格朗日乘子。考虑以下量：

<div class="math">
$$
\theta_\mathcal{P}(w) = \max_{\alpha, \beta:\alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta)
$$
</div>

这里，<span class="math">$ \mathcal{P} $</span> 代表 "原问题"。让我们假设一个特定的 <span class="math">$w$</span> 满足原问题的约束条件（即，如果 <span class="math">$g_i(w) > 0$</span> 或 <span class="math">$h_i(w) \neq 0$</span> 对某些 <span class="math">$ i $</span>）。那么有：

<div class="math">
$$
\begin{align*}
\theta_\mathcal{P}(w) &= \max_{\alpha, \beta; \alpha_i \geq 0} f(w) + \sum_{i=1}^{k} \alpha_i g_i(w) + \sum_{i=1}^{l} \beta_i h_i(w) \tag{1} \\[5pt]
&= \infty \tag{2}
\end{align*}
$$
</div>

相反，如果一个特定的 <span class="math">$w$</span> 满足约束条件，那么 <span class="math">$\theta_\mathcal{P}(w) = f(w)$</span>。就有：

<div class="math">
$$
\begin{align*}
\theta_\mathcal{P}(w) = \begin{cases}
f(w) & \text{if } w \text{ satisfies primal constraints} \\[5pt]
\infty & \text{otherwise}
\end{cases}
\end{align*}
$$
</div>

因此，<span class="math">$\theta_\mathcal{P}(w)$</span> 在我们的问题中对于所有满足原始约束的 <span class="math">$w$</span> 值都是相同的，并且如果约束条件被违反，则会是正无穷。所以，我们考虑下面的最小化问题：

<div class="math">
$$
\min_{w} \ \theta_\mathcal{P}(w) = \min_{w} \ \max_{\alpha, \beta} \mathcal{L}(w, \alpha, \beta)
$$
</div>

我们看到这是相同的问题，即与原始问题一样，并且有相同的解，我们将原始问题的最优值定义为 <span class="math">$p^* = \text{min} \ \theta_\mathcal{P}(w)$</span>；这是原始问题的目标值。

现在，让我们看一个稍微不同的问题。我们定义：

<div class="math">
$$
\theta_\mathcal{D}(\alpha, \beta) = \min_{w} \mathcal{L}(w, \alpha, \beta)
$$
</div>

这里，<span class="math">$ \mathcal{D} $</span> 代表 "对偶"。注意在 <span class="math">$\theta_\mathcal{D}$</span> 的定义中我们是在最小化关于 <span class="math">$ w $</span> 而最大化关于 <span class="math">$\alpha, \beta$</span>，这里我们是在最小化关于这些参数的问题：

<div class="math">
$$
\max_{\alpha, \beta:\alpha_i \geq 0} \ \theta_\mathcal{D}(\alpha, \beta) = \max_{\alpha, \beta:\alpha_i \geq 0} \ \min_{w} \mathcal{L}(w, \alpha, \beta)
$$
</div>

这完全是同样的问题，只是 "max" 和 "min "的顺序被交换了。我们定义对偶问题的最优值为 <span class="math">$d^* = \max_{\alpha,\beta}\ \theta_\mathcal{D}(\alpha, \beta)$</span>。

如何证明原问题和对偶问题之间的关系？可以容易地显示出：

<div class="math">
$$
d^* = \max_{\alpha, \beta; \alpha_i \geq 0} \min_w \mathcal{L}(w, \alpha, \beta) \leq \min_w \max_{\alpha, \beta; \alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta) = p^*
$$
</div>

在某些约束条件下，我们会有 <span class="math">$p^* = d^*$</span>。

#### 对偶强弱

##### 弱对偶

所谓弱对偶性，指的是 Lagrange 对偶问题的一个性质：

<div class="math">
$$
d^* \leq p^*
$$
</div>

下面是弱对偶的示意图：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/07/15/pk5oLyn.png" data-lightbox="image-6" data-title="weak duality">
  <img src="https://s21.ax1x.com/2024/07/15/pk5oLyn.png" alt="weak duality" style="width:100%;max-width:500px;cursor:pointer">
 </a>
</div>

即便原问题不是凸问题，上述不等式也成立，这就是 **弱对偶性（Weak Duality）**。这种性质即使是 <span class="math">$d^*, p^*$</span> 为无穷也成立:

* 如果 <span class="math">$p^* = -\infty$</span>，则 <span class="math">$d^* = -\infty$</span>
* 如果 <span class="math">$d^* = +\infty$</span>，则 <span class="math">$p^* = +\infty$</span>

我们还定义 <span class="math">$p^* - d^*$</span> 称为 **最优对偶间隙（Optimal Duality Gap）**，这个值必然非负。

##### 强对偶

当 <span class="math">$ d^* = p^*$</span> 时，称为 **强对偶性（Strong Duality）**。

下面是强对偶的示意图：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/07/15/pk5TPSJ.png" data-lightbox="image-6" data-title="strong duality">
  <img src="https://s21.ax1x.com/2024/07/15/pk5TPSJ.png" alt="strong duality" style="width:100%;max-width:500px;cursor:pointer">
 </a>
</div>

显然，强对偶需要满足一定的限制条件。

#### KKT 条件

在之前的假设下，必须存在 <span class="math">$w^*$</span>, <span class="math">$\alpha^*$</span>, <span class="math">$\beta^*$</span> 使得 <span class="math">$w^*$</span> 是原问题的解，<span class="math">$\alpha^*$, $\beta^*$</span> 是对偶问题的解。此外，<span class="math">$p^* = d^* = \mathcal{L}(w^*, \alpha^*, \beta^*)$</span>。同时，<span class="math">$w^*$</span>, <span class="math">$\alpha^*$</span> 和 <span class="math">$\beta^*$</span> 满足  **Karush-Kuhn-Tucker（KKT）条件**，具体如下：

<div class="math">
$$
\begin{align*}
\frac{\partial}{\partial w_i} \mathcal{L}(w^*, \alpha^*, \beta^*) &= 0, \quad i = 1, \ldots, n \quad \tag{3} \\[5pt]
\frac{\partial}{\partial \beta_i} \mathcal{L}(w^*, \alpha^*, \beta^*) &= 0, \quad i = 1, \ldots, l \quad \tag{4} \\[5pt]
\alpha_i^* g_i(w^*) &= 0, \quad i = 1, \ldots, k \quad \tag{5} \\[5pt]
g_i(w^*) &\leq 0, \quad i = 1, \ldots, k \quad \tag{6} \\[5pt]
\alpha_i^* &\geq 0, \quad i = 1, \ldots, k \quad \tag{7}
\end{align*}
$$
</div>

如果某个 <span class="math">$w^*$</span>, <span class="math">$\alpha^*$</span>, <span class="math">$\beta^*$</span> 满足 KKT 条件，则它们同时也是原问题和对偶问题的解。

我们特别指出公式（5），这是所谓的 KKT 对偶互补条件。具体来说，如果 <span class="math">$\alpha_i^* > 0$</span>，则 <span class="math">$g_i(w^*) = 0$</span>。（即，<span class="math">$g_i(w) \leq 0$</span> 约束是活跃的，意味着它以等式而不是不等式的形式成立。）后续，这将是证明 SVM 中仅有少数 "支持向量" 的关键；KKT 对偶互补条件也将在我们讨论 SMO 算法时，为我们提供收敛性测试。

### 最优边界决策器

在先前的讨论中，我们希望能够找到一个决策边界，这个（几何）边界需要尽可能的大——越大的边界意味着越高的置信度。更形象一些，（几何）边界会像一个巨大的鸿沟，将正负样本分隔在边界的两侧。

现在，我们假设训练集中的样例都是线性可分的——表示为这些样本可由某些超平面来分隔开。为找出最大的（几何）边界，我们给出如下的表示：

<div class="math">
$$
\begin{aligned}
\max&_{\gamma,w,b}\  \gamma \\[5pt]
\text{s.t.}& \quad \quad y^{(i)}(w^T x^{(i)}+b) \geq \gamma,\quad i = 1,2,...,m \\[5pt]
&\quad \quad ||w||=1
\end{aligned}
$$
</div>

在上述条件中，通过 <span class="math">$ ||w|| = 1 $</span> 使得函数边界等于几何边界，从而将两者联系起来。同时，保证了几何边界的最小值是 <span class="math">$\gamma$</span>。解决上述问题将得到 <span class="math">$ (w, b) $</span> 相对于训练集 <span class="math">$ S $</span> 的最大几何边界。

但是，<span class="math">$ ||w|| = 1 $</span> 是一个非凸约束，不容易直接解出上述问题。所以，考虑将问题转化为另一个形式：

<div class="math">
$$
\begin{aligned}
\max&_{\hat{\gamma},w,b}\ \ \frac{\hat{\gamma} }{||w||}\\[5pt]
\text{s.t.}& \quad \quad \ y^{(i)}(w^T x^{(i)}+b) \geq \hat{\gamma},\quad i = 1,2,...,m
\end{aligned}
$$
</div>

很不幸，<span class="math">$\displaystyle{\frac{\hat{\gamma} }{||w||}}$</span> 仍然是非凸的。继续将问题进行转化。

在之前的对于边界的讨论中，我们提到函数边界可以随 <span class="math">$ (w, b) $</span> 进行任意的放缩。于是，在这里我们规定，<span class="math">$ (w, b) $</span> 相对于 <span class="math">$ S $</span> 的函数边界为 1 ：

<div class="math">
$$
\hat{\gamma} = 1
$$
</div>

在这样规定后，问题被等价转化为了：

<div class="math">
$$
\begin{aligned}
\min&_{\gamma,w,b}\ \ \frac{1}{2}||w||^2\\[5pt]
\text{s.t.}& \quad \quad \ y^{(i)}(w^T x^{(i)}+b) \geq 1,\quad i = 1,2,...,m
\end{aligned}
$$
</div>

显然，<span class="math">$\displaystyle{\frac{1}{2}||w||^2}$</span> 是一个带有凸二次目标函数和线性约束的优化问题。它的解给我们提供了最优边界分决策器。这种优化问题可以使用常见的 **二次规划（quadratic programming）** 算法来解决。

在基本了解了最优边界决策器后，我们将通过我们之前了解到的拉格朗日对偶来进一步深入。

将约束转化为：

<div class="math">
$$
g_i(w) = -y^{(i)}(w^T x^{(i)}+b) + 1 \leq 0
$$
</div>

由 KKT 条件的等式<span class="math">$(5)$</span>与等式<span class="math">$(7)$</span>得：

拉格朗日算子 <span class="math">$ \alpha_i \geq 0 $</span> 。若要使 <span class="math">$ \alpha_i \cdot g_i(w) = 0 $</span>，则有 <span class="math">$ \alpha_i = 0 $</span> 或 <span class="math">$ g_i(w) = 0 $</span>。

当 <span class="math">$ \alpha_i > 0 $</span> 时，且仅当 <span class="math">$ g_i(w) = 0 $</span> 满足条件。即 <span class="math">$ -y^{(i)}(w^T x^{(i)}+b) + 1 = 0$</span>。自然得到：<span class="math">$y^{(i)}(w^T x^{(i)}+b) = 1$</span>，即 <span class="math">$ \hat{\gamma} = 1$</span>。满足上述条件的样本点，一定是距离决策边界最近的，如下图所示：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/07/15/pk5H1dP.png" data-lightbox="image-6" data-title="support vectors">
  <img src="https://s21.ax1x.com/2024/07/15/pk5H1dP.png" alt="support vectors" style="width:100%;max-width:400px;cursor:pointer">
 </a>
</div>

这些在虚线上，距离决策边界最近的点被称为 **支持向量（support vector）**。支持向量的特性是函数边界 <span class="math">$ \hat{\gamma} = 1$ 且其拉格朗日算子 <span class="math">$ \alpha_i > 0 $。

当 <span class="math">$ \alpha_i = 0 $</span>，此时单独考虑 <span class="math">$ g_i(w) = -y^{(i)}(w^T x^{(i)}+b) + 1 < 0 $</span>。对其进行等价转换得 <span class="math">$ g_i(w) = y^{(i)}(w^T x^{(i)}+b) > 1 $</span>。表示这些点未越过由支持向量构成得决策平面，它们被正确地分类。

接下来我们对上述优化问题的对偶问题进行讨论。

当我们尝试利用拉格朗日对偶来解决此问题时，其中一个关键的点在于只用输入特征的内积 <span class="math">$ \langle x^{(i)}, x^{(j)} \rangle $</span> 来表示算法。

> [!NOTE]
> **<span class="math">$ \langle x^{(i)}, x^{(j)} \rangle $</span>** 等价于 **<span class="math">$ (x^{(i)})^T x^{(j)} $</span>。**

当我们为我们的对偶优化问题构建拉格朗日函数时，我们有：

<div class="math">
$$
\begin{align*}
\mathcal{L}(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^m \alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1] \tag{8}
\end{align*}
$$
</div>
注意这里只有 <span class="math">$\alpha_i$</span>，而没有 <span class="math">$\beta_i$</span>。表明拉格朗日乘子只有不等式约束。

接下来，需要找到问题的对偶形式。为此，我们首先需要对 <span class="math">$w$</span> 和 <span class="math">$b$</span>（对于固定的 <span class="math">$\alpha$</span>）求导并令其等于零以最小化 <span class="math">$\mathcal{L}(w, b, \alpha)$</span>，进而得到 <span class="math">$θ_\mathcal{D}$</span>。于是，我们有：

<div class="math">
$$
\nabla_w L(w, b, \alpha) = w - \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)} = 0
$$
</div>

这意味着：

<div class="math">
$$
\begin{align*}
w = \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)} \tag{9}
\end{align*}
$$
</div>

其对于 <span class="math">$b$</span> 的导数为：

<div class="math">
$$
\frac{\partial}{\partial b} L(w, b, \alpha) = \sum_{i=1}^m \alpha_i y^{(i)} = 0 \tag{10}
$$
</div>

如果取 <span class="math">$w$</span> 在方程<span class="math">$(9)$</span>中的定义并将其重新代入拉格朗日方程（方程<span class="math">$(8)$</span>），并化简，会得到：

<div class="math">
$$
L(w, b, \alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j x^{(i)T} x^{(j)} - b \sum_{i=1}^m \alpha_i y^{(i)}
$$
</div>

又根据方程<span class="math">$(10)$</span>，得到最后一项必须为零，因此有：

<div class="math">
$$
L(w, b, \alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j (x^{(i)})^T x^{(j)}
$$
</div>

由于我们是通过最小化 $ \mathcal{L}$ 相对于 <span class="math">$w$</span> 和 <span class="math">$b$</span> 的函数得到上述方程的，并且一直有 <span class="math">$\alpha_i \geq 0 $</span> 的约束和方程<span class="math">$(10)$</span>。于是，我们得到以下对偶优化问题：

<div class="math">
$$
\begin{aligned}
\text{max}_\alpha \quad &W(\alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle \\[5pt]
\text{s.t.}\quad &\alpha_i ≥ 0, i = 1, \ldots, m \\[5pt]
&\sum_{i=1}^m \alpha_i y^{(i)} = 0
\end{aligned}
$$
</div>

您也应该能够验证 <span class="math">$p^* = d^*$</span> 和 KKT 条件（方程 3-7）确实适用于我们的优化问题。因此，我们可以解决对偶问题而不是原始问题。具体来说，在上述的对偶问题中，我们有一个最大化问题，其参数是 <span class="math">$\alpha_i$</span>。我们稍后再谈这个问题。

关于我们将用于解决对偶问题的特定算法的讨论可以留到后面，但如果我们确实能解决它（即，找到最大化 <span class="math">$W(\alpha)$</span> 的 <span class="math">$\alpha$</span> 值，并满足约束条件，那么我们可以使用方程<span class="math">$(9)$</span>回去找到 <span class="math">$\alpha$</span> 的函数作为最优 <span class="math">$w$</span> 的解。找到 <span class="math">$w^*$</span> 后，通过考虑原始问题，找到截距项 <span class="math">$b$</span> 的最优值也很直接。具体而言，截距项 <span class="math">$b$</span> 可以表示为：

<div class="math">
$$
b^* = -\frac{\max_{y^{(i)}=1} w^{*T}x^{(i)} + \min_{y^{(i)}=-1} w^{*T}x^{(i)}}{2} \tag{11}
$$
</div>

在继续之前，让我们也仔细看看方程<span class="math">$(9)$</span>，它给出了最优 <span class="math">$w$</span> 的值，即 <span class="math">$\alpha$</span> 的最优值的函数。假设我们已经将我们模型的参数适配到一个训练集上，现在希望对一个新的输入点 <span class="math">$x$</span> 进行预测。我们将计算 <span class="math">$w^T x + b$</span> 并且只有当这个量大于零时，才预测 <span class="math">$y = 1$</span>。但是使用方程<span class="math">$(9)$</span>，这个量也可以写成：

<div class="math">
$$
\begin{align*}
w^T x + b = \left( \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)} \right)^T x + b \tag{12} \\[5pt]
= \sum_{i=1}^m \alpha_i y^{(i)} \left( x^{(i)} \right)^T x + b \tag{13}
\end{align*}
$$
</div>

因此，如果我们找到了 <span class="math">$\alpha_i$</span>，为了做出预测，我们需要计算一个只依赖于 <span class="math">$x$</span> 和所有训练集数据点之间内积的量。我们之前已经看到 <span class="math">$\alpha_i$</span> 将全部为零，除了支持向量。因此，许多在总和上述的项将为零，而我们真正需要考虑的只是内积在 <span class="math">$x$</span> 和支持向量之间的（通常只有少数几个）来计算<span class="math">$(13)$</span>并做出我们的预测。

通过检查问题的对偶形式，我们获得了问题结构的重要见解，并且还能够完全用内积项将整个算法表达出来，仅应用到特征向量。在下一节中，我们将利用这个属性应用内核技巧到我们的分类问题中。由此推导出的算法——支持向量机，将能够在非常高维的空间中进行有效地学习。

### 核 Kernel

在之前关于线性回归的讨论中，考虑输入的 <span class="math">$ x $</span>，与之对应的拟合曲线使用特征 <span class="math">$ x, x^2, x^3 $</span> 得到了一个三次函数。为了区分这两组变量，我们将原始输入值称为问题的输入 **属性（attributes）**（在这个例子中，即为 <span class="math">$ x $</span>）。当这些属性映射到一些新的量，然后传递给学习算法时，我们将这些新的量称为输入 **特征（features）**。<span class="math">$ \phi $</span> 表示 **特征映射（feature mapping）**，它将属性映射到特征。

如上例子，我们有：

<div class="math">
$$
\phi(x) = \begin{bmatrix}
\phantom{*} x \phantom{*} \\
\phantom{*} x^2 \phantom{*} \\
\phantom{*} x^3 \phantom{*} \\
\end{bmatrix}
$$
</div>

相较于使用 SVM 的原始输入属性 <span class="math">$ x $</span> 应用于学习算法，我们可能想要使用一些特征 <span class="math">$ \phi(x) $</span> 来学习。由于，算法完全可以用内积的形式表达，这意味着我们可以用 <span class="math">$ \phi(x), \phi(z) $</span> 替换所有的内积。为此，我们只需回顾之前的算法，并用 <span class="math">$ \phi(x) $</span> 替换其中的 <span class="math">$ x $</span>。具体来说，给定一个特征映射 <span class="math">$ \phi $</span>，我们定义相应的 **核（kernel）** 为：

<div class="math">
$$
K(x, z) = \phi(x)^T \phi(z)
$$
</div>

然后，在我们之前算法中使用的任何地方都用 <span class="math">$ K(x, z) $</span> 替换 <span class="math">$ \langle x, z \rangle $</span>，现在我们的算法将使用特征 <span class="math">$ \phi $</span> 学习。现在，给定 <span class="math">$ \phi $</span>，我们可以通过找到 <span class="math">$ \phi(x) $</span> 和 <span class="math">$ \phi(z) $</span> 并取它们的内积来轻松计算 <span class="math">$ K(x, z) $</span>。但值得注意的是，尽管 <span class="math">$ \phi(x) $</span> 本身可能计算成本很高（因为其可能为一个非常高维度的向量），<span class="math">$ K(x, z) $</span> 的计算通常很简易。在这种情况下，通过在算法中使用一个有效的方式来计算 <span class="math">$ K(x, z) $</span>，我们可以让 SVM 在由 <span class="math">$ \phi $</span> 给定的高维特征空间中学习，而不需要显式地找到或表示向量 <span class="math">$ \phi(x) $</span>。

下面是一个例子。假设 <span class="math">$ x, z \in \mathbb{R}^n $</span>，并考虑

<div class="math">
$$
K(x, z) = (x^T z)^2
$$
</div>

我们还可以将其写成以下形式：

<div class="math">
$$
\begin{aligned}
K(x, z) &= \left(\sum_{i=1}^n x_iz_i\right) \left(\sum_{j=1}^n x_iz_i\right) \\[5pt]
&= \sum_{i=1}^n \sum_{j=1}^n x_iz_ix_jz_j \\[5pt]
&= \sum_{i,j=1}^n (x_ix_j)(z_iz_j)
\end{aligned}
$$
</div>

因此，我们看到 <span class="math">$ K(x, z) = \phi(x)^T \phi(z) $</span>，其中特征映射 <span class="math">$ \phi $</span> 给出如下（这里展示的是 <span class="math">$ n = 3 $</span> 的情况）:

<div class="math">
$$
\phi(x) = \begin{bmatrix}
\phantom{*} x_1x_1 \phantom{*} \\
\phantom{*} x_1x_2 \phantom{*} \\
\phantom{*} x_1x_3 \phantom{*} \\
\phantom{*} x_2x_1 \phantom{*} \\
\phantom{*} x_2x_2 \phantom{*} \\
\phantom{*} x_2x_3 \phantom{*} \\
\phantom{*} x_3x_1 \phantom{*} \\
\phantom{*} x_3x_2 \phantom{*} \\
\phantom{*} x_3x_3 \phantom{*}
\end{bmatrix}
$$
</div>

请注意，尽管计算高维度的 <span class="math">$ \phi(x) $</span> 需要 <span class="math">$ O(n^2) $</span> 时间，但找到 <span class="math">$ K(x, z) $</span> 只需要 <span class="math">$ O(n) $</span> 时间——在输入属性的维度上是线性的。

对于相关的核，考虑以下形式：

<div class="math">
$$
\begin{aligned}
K(x, z) &= (x^\top z + c)^2 \\[5pt]
&= \sum_{i=1}^n (x_ix_j)(z_iz_j) + \sum_{i=1}^n \sqrt{2cx_i} \sqrt{2cz_i} + c^2
\end{aligned}
$$
</div>

其对应的特征映射如下（<span class="math">$ n = 3 $</span>）：

<div class="math">
$$
\phi(x) = \begin{bmatrix}
\phantom{*} x_1x_1 \phantom{*} \\
\phantom{*} x_1x_2 \phantom{*} \\
\phantom{*} \vdots \phantom{*} \\
\phantom{*} x_3x_3 \phantom{*} \\
\phantom{*} \sqrt{2cx_1} \phantom{*} \\
\phantom{*} \vdots \phantom{*} \\
\phantom{*} \sqrt{2cx_3} \phantom{*} \\
\phantom{*} c \phantom{*}
\end{bmatrix}
$$
</div>

参数 <span class="math">$ c $</span> 控制了 <span class="math">$ x_i $</span>（一阶项）和 <span class="math">$ x_ix_j $</span>（二阶项）之间的相对权重。

进一步推广，核 <span class="math">$ K(x, z) = (x^T z + c)^d $</span> 对应于一个特征映射到一个 <span class="math">$ \small \begin{pmatrix} n+d \\\\ d \end{pmatrix} $</span> 维特征空间，涵盖了所有的 <span class="math">$ x_{i1}, x_{i2} \cdots x_{ik} $</span> 到 <span class="math">$ d $</span> 阶的单项式。然而，尽管在这个 <span class="math">$ n^d $</span> 维空间中操作，计算 <span class="math">$ K(x, z) $</span> 仍然只需要 <span class="math">$ O(n^2) $</span> 时间，因为我们不需要显式地表示这个非常高维的特征向量。

现在，让我们从另一个稍微不同的角度来思考核的概念。直观上，如果 <span class="math">$ \phi(x) $ 和 <span class="math">$ \phi(z) $ 非常接近，则 <span class="math">$ K(x, z) = \phi(x)^\top \phi(z) $ 应该很大。相反，如果 <span class="math">$ \phi(x) $ 和 <span class="math">$ \phi(z) $ 非常远离——比如几乎正交——那么 <span class="math">$ K(x, z) $ 将会很小。

因此，我们可以将 <span class="math">$ K(x, z) $</span> 看作是 <span class="math">$ \phi(x) $</span> 和 <span class="math">$ \phi(z) $</span> 的相似度或者接近度的度量。

基于上述的考量，假设你正在处理某个学习问题，并且你想到了一个你认为可能是 <span class="math">$ x $</span> 和 <span class="math">$ z $</span> 相似度的合理度量的函数 <span class="math">
$ K(x, z) $</span> 如下：

<div class="math">
$$
K(x, z) = \exp\left(-\frac{\|x-z\|^2}{2\sigma^2}\right)
$$
</div>

这是一个衡量 <span class="math">$ x $</span> 和 <span class="math">$ z $</span> 相似度的合理度量，当 <span class="math">$ x $</span> 和 <span class="math">$ z $</span> 接近时接近 1，当 <span class="math">$ x $</span> 和 <span class="math">$ z $</span> 远离时接近 0。我们可以使用这个来定义核。这个核也被称为 **高斯核（Gaussian kernel）** （对应到一个无限维的特征映射 <span class="math">$ \phi $</span>）。

但更广泛地说，给定某个函数 <span class="math">$ K $</span>，我们如何判断它是否是一个有效的核函数；也就是说，我们是否可以判断是否存在某个特征映射 <span class="math">$ \phi $</span>，使得 <span class="math">$ K(x, z) = \phi(x)^T\phi(z) $</span> 对所有 <span class="math">$ x, z $</span> 成立？

现在假设 <span class="math">$ K $</span> 确实是对应于某个特征映射 <span class="math">$ \phi $</span> 的有效核函数。现在，考虑一些有限集合的点（不一定是训练集） <span class="math">$ \{x^{(1)}, \ldots, x^{(m)}\} $</span>，并让一个方形的 <span class="math">$ m \times m $</span> 矩阵 <span class="math">$ K $</span> 被定义为其 <span class="math">$ (i, j) $</span>-项由 <span class="math">$ K_{ij} = K(x^{(i)}, x^{(j)}) $</span> 给出。这个矩阵被称为 **核矩阵（kernel matrix）**。注意，我们在这里重载了符号 <span class="math">$ K $</span>，用它同时表示核函数 <span class="math">$ K(x, z) $</span> 和核矩阵 <span class="math">$ K $</span>，这是由于它们显而易见的关系。

现在，如果 <span class="math">$ K $</span> 是一个有效的核函数，那么有 <span class="math">$ K_{ij} = K(x^{(i)}, x^{(j)}) = \phi(x^{(i)})^T \phi(x^{(j)}) = K(x^{(i)}, x^{(j)}) = K_{ji} $</span>，因此 <span class="math">$ K $</span> 必须是对称的。此外，设 <span class="math">$ \phi_k(x) $</span> 表示向量 <span class="math">$ \phi(x) $</span> 的第 <span class="math">$ k $</span> 个坐标，对于任何向量 <span class="math">$ z $</span>，我们有：

<div class="math">
$$
\begin{aligned}
z^T K z &= \sum_i \sum_j z_i K_{ij} z_j \\[5pt]
&= \sum_i \sum_j z_i \phi(x^{(i)})^T \phi(x^{(j)}) z_j \\[5pt]
&= \sum_i \sum_j z_i \sum_k \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j \\[5pt]
&= \sum_k \sum_i \sum_j z_i \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j \\[5pt]
&= \sum_k \left( \sum_i z_i \phi_k(x^{(i)}) \right)^2 \\[5pt]
&\geq 0
\end{aligned}
$$
</div>

由于 <span class="math">$ z $</span> 是任意的，这表明 <span class="math">$ K $</span> 是正半定的（<span class="math">$ K \geq 0 $</span>）。

现在，我们已经证明，如果 <span class="math">$ K $</span> 是一个有效的核（即，如果它对应于某个特征映射 <span class="math">$ \phi $</span>），那么相应的核矩阵 <span class="math">$ K \in \mathbb{R}^{m \times m} $</span> 是对称正半定的。更一般地，这不仅是 <span class="math">$ K $</span> 成为有效核的必要条件，而且是充分条件（也称为 Mercer 核）。以下结果是由于 Mercer 而来的。

许多文献以稍微复杂的形式展示 Mercer 定理，涉及 <span class="math">$ L^2 $</span> 函数，但当输入属性取 <span class="math">$ \mathbb{R}^n $</span> 中的值时，这里给出的版本是等效的。

上述倒数第二步使用了之前问题集中的同一个技巧。由于 <span class="math">$ z $</span> 是任意的，这表明 <span class="math">$ K $</span> 是半正定的。

因此，如果 <span class="math">$ K $</span> 是一个有效的核（即它对应于某个特征映射 <span class="math">$ \phi $</span>），那么相应的核矩阵 <span class="math">$ K $</span> 是对称正定的。更一般地说，这不仅是 <span class="math">$ K $</span> 是有效核的必要条件，也是充分条件（也称为 Mercer 核）。

> [!TIP]
> <font size=4>**Mercer 定理**</font><br>
> 设 <span class="math">$ K : \mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R} $</span> 给定。那么对于 <span class="math">$ K $</span> 成为一个有效的（Mercer）核函数，必要且充分的条件是对于任何 <span class="math">$ \\{x^{(1)}, \ldots, x^{(m)}\\} $</span>（<span class="math">$ m < \infty $</span>），相应的核矩阵 <span class="math">$ K $</span> 是对称正半定的。

给定一个函数 <span class="math">$ K $</span>，除了尝试找到对应的特征映射 <span class="math">$ \phi $</span> 外，这个定理因此提供了另一种测试它是否为有效核函数的方法。

核函数在支持向量机中的应用就暂告一个段落。但是，请记住，核函数的概念在应用上比 SVM 更广泛。具体来说，如果你有任何学习算法，可以用输入属性向量之间的内积 <span class="math">$ \langle x, z \rangle $</span> 表示，那么通过用 <span class="math">$ K(x, z) $</span> 替换这个内积（其中 <span class="math">$ K $</span> 是一个核函数）。这样你的算法就可以在核函数所对应的高维特征空间中有效地工作。例如，这个核技巧可以应用于感知机，导出核感知机算法。后面我们将看到的许多算法也会运用这个思想，这种处理方法称为 **核技巧（kernel trick）**。

### 正则化

到目前为止，有关 SVM 的推导，都是在假设数据是线性可分的前提下进行的。虽然通过 <span class="math">$\phi$</span> 将数据映射到高维特征空间可以增加数据可分的可能性，我们不能保证总是可以通过维度提升来解决这个问题。此外，在某些情况下，由于某些离群数据，我们也无法明确找到一个完全令人满意的分隔超平面。例如，下面显示了一个决策边界，当添加一个离群值时，它导致决策边界发生剧烈转变，使得其显著缩小。

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/08/09/pASSCCR.png" data-lightbox="image-6" data-title="outlier">
  <img src="https://s21.ax1x.com/2024/08/09/pASSCCR.png" alt="outlier" style="width:100%;max-width:800px;cursor:pointer">
 </a>
</div>

为了让算法同样适用于非线性可分数据集并且对离群值不过于敏感，这里重新描述我们的优化问题（使用 <span class="math">$ \mathcal{l}_1 $</span> **正则化（regularization）**）如下：

<div class="math">
$$
\begin{aligned}
\min&_{w,b,\xi}\quad \frac{1}{2}\|w\|^2 + C \sum_{i=1}^m \xi_i \\[5pt]
&\text{s.t. }\quad y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i , \ i = 1, \ldots, m \\[5pt]
&\quad\quad\ \  \xi_i \geq 0, \ i = 1, \ldots, m
\end{aligned}
$$
</div>

借由正则化后，现在样本被允许有小于 1 的（函数）边界，如果一个样本的函数边界为 <span class="math">$ 1 - \xi_i$</span>（<span class="math">$\xi_i \geq 0$</span>），我们将为目标函数增加 <span class="math">$C\xi_i$</span> 的额外代价。<span class="math">$C$</span> 是惩罚系数。如果 <span class="math">$y^{(i)}(w^T x^{(i)} + b) < 1 - \xi_i$</span>，则损失增加 <span class="math">$\xi_i$</span>，这是乘以常数 <span class="math">$C$</span>，并且加到目标函数中，这就允许一些样本点距离间隔的边界有一定的违背程度，从而使模型能更好地泛化。

如前所述，重写广义拉格朗日函数：

<div class="math">
$$
L(w,b,\xi,\alpha,r) = \frac{1}{2}w^Tw + C \sum_{i=1}^m \xi_i - \sum_{i=1}^m \alpha_i [y^{(i)}(w^T\phi(x^{(i)})+b) - 1 + \xi_i] - \sum_{i=1}^m r_i\xi_i
$$
</div>

这里，<span class="math">$\alpha_i$</span> 和 <span class="math">$r_i$</span> 是我们的拉格朗日乘子（限制为 <span class="math">$\geq 0$</span>）。将 <span class="math">$w$</span> 和 <span class="math">$b$</span> 的导数设置为零后，代回并简化，我们得到了如下的对偶形式问题：

<div class="math">
$$
\begin{aligned}
\text{max}_\alpha \quad &W(\alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m y^{(i)}y^{(j)}\alpha_i\alpha_j\langle x^{(i)}, x^{(j)} \rangle \\[5pt]
\text{s.t. }\quad &0 \leq \alpha_i \leq C, \ i = 1, \ldots, m \\[5pt]
&\sum_{i=1}^m \alpha_i y^{(i)} = 0
\end{aligned}
$$
</div>

类似地，如方程<span class="math">$(9)$</span>，我们可以将 <span class="math">$w$</span> 利用 <span class="math">$\alpha_i$</span> 来表示，从而在解决对偶问题后，我们可以继续使用方程(13)进行预测。注意，在添加 <span class="math">$ \mathcal{l}_1 $</span> 正则化后，对偶问题唯一的变化是：原来的约束 <span class="math">$0 \leq \alpha_i$</span> 现在变成了 <span class="math">$0 \leq \alpha_i \leq C$</span>；同时，对 <span class="math">$b$</span> 的计算也必须修改（方程<span class="math">$(11)$</span>现在不再有效）。

之前所给出的 KKT 对偶互补条件，也是相应转变为：

<div class="math">
$$
\begin{align*}
\alpha_i = 0 \quad &\Rightarrow \quad y^{(i)}(w^T\phi(x^{(i)})+b) \geq 1 \tag{14} \\[5pt]
\alpha_i = C \quad &\Rightarrow \quad y^{(i)}(w^T\phi(x^{(i)})+b) \leq 1 \tag{15} \\[5pt]
0 < \alpha_i < C \quad &\Rightarrow \quad y^{(i)}(w^T\phi(x^{(i)})+b) = 1 \tag{16}
\end{align*}
$$
</div>

现在，我们只需要提供一个解决对偶问题的算法，在下一节中进行详解。

### SMO 算法

**SMO 算法 （sequential minimal optimization）**，即序列最小化优化算法。在1998年由 John Platt 提出，是一种用于解决支持向量机训练期间出现的二次规划问题的算法。再进一步学习 SMO 前，我们需要先了解一下坐标上升优化算法。

#### 坐标上升

考虑一个无约束优化问题：

<div class="math">
$$
\begin{align*}
\text{max}_\alpha \ W(\alpha_1, \alpha_2, \dots, \alpha_m)
\end{align*}
$$
</div>

这里，我们将 <span class="math">$W$</span> 视为参数 <span class="math">$\alpha_i$</span> 的某个函数，先暂时忽略这个问题和 SVM 之间的关系。之前我们已经学习过两种优化算法：梯度上升与牛顿法。这里将要学习的新算法称为 **坐标上升（coordinate ascent）**，具体如下：

<div class="math">
$$
\begin{align*}
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ &\mathrm{Loop\ until\ convergence:}\ \{ \\[5pt]
&\ \ \ \ \ \ \mathrm{for}\ i = 1, \cdots,m,\ \{ \\[5pt]
&\ \ \ \ \ \ \ \ \ \ \ \ \alpha_i = \arg\text{max}_{\hat{\alpha_i}}\ W(\alpha_1, \dots, \alpha_{i-1}, \hat{\alpha_i}, \alpha_{i+1}, \dots, \alpha_m)\\
&\ \ \ \ \ \ \} \\
&\}
\end{align*}
$$
</div>

坐标上升算法的核心思想是：虽然同时优化多个变量可能较为复杂，但优化单个变量通常会相对简单。通过固定其他所有变量，仅对当前选择的单个变量进行优化，我们可以简化每一步的计算过程，从而逐步接近全局最优解或局部最优解。

因此，在这个算法的最内层循环中，我们将固定某些 <span class="math">$\alpha_i$</span> 并仅重新优化关于 <span class="math">$\alpha_i$</span> 的 <span class="math">$W$</span>。在此方法的这一版本中，内循环按 <span class="math">$\alpha_1, \alpha_2, \dots, \alpha_m, \alpha_1, \alpha_2, \dots$</span> 的顺序优化变量。（一个更复杂的版本可能会选择其他排序；例如，我们可能会选择的下一个要更新的变量，是基于我们期望它能使 <span class="math">$W$</span>(<span class="math">$\alpha$</span>)最大增加的预期。）

当函数 <span class="math">$W$</span> 的形式使得 "<span class="math">$\text{arg max}$</span>" 在内层循环中能够高效执行时，坐标上升可以是一个相当高效的算法。这里是坐标上升实际操作的示意图：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/08/10/pASrgJ0.png" data-lightbox="image-6" data-title="coordinate ascent">
  <img src="https://s21.ax1x.com/2024/08/10/pASrgJ0.png" alt="coordinate ascent" style="width:100%;max-width:525px;cursor:pointer">
 </a>
</div>

图中的椭圆是我们希望优化的二次函数的等高线。坐标上升算法从 <span class="math">$(2, -2)$</span> 开始初始化，并且图中也描绘了它前往全局最大值的路径。请注意，在每一步中，坐标上升都会沿着与坐标轴平行的方向前进一步，因为每次只优化一个变量。

#### SMO 详解

基于前面的讲解，我们得到新的（对偶）优化问题如下：

<div class="math">
$$
\begin{align*}
\text{max}_\alpha\ &W(\alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x_i, x_j \rangle \tag{17} \\[5pt]
\text{s.t.}\ &0 \leq \alpha_i \leq C,\quad i = 1, \ldots, m \tag{18} \\[5pt]
&\sum_{i=1}^m \alpha_i y^{(i)}  = 0 \tag{19}
\end{align*}
$$
</div>

假设我们已经设置对应的的一组 <span class="math">$\alpha_i$</span> 满足约束条件（18-19） 。现在，假设我们想固定 <span class="math">$\alpha_2, \ldots, \alpha_m$</span>，并对目标函数关于 <span class="math">$\alpha_1$</span> 进行坐标上升算法来重新优化。但是似乎存在问题，因为约束条件（19）确保了

<div class="math">
$$
\alpha_1 y^{(1)} = -\sum_{i=2}^m \alpha_i y^{(i)}
$$
</div>

或者，将两边乘以 <span class="math">$y^{(1)}$</span>，我们等价地有

<div class="math">
$$
\alpha_1 = -y^{(1)}\sum_{i=2}^m \alpha_i y^{(i)}
$$
</div>

> [!NOTE]
> 由于 <span class="math">$y^{(1)} \in \\{-1, 1\\}$</span>，有 <span class="math">$(y^{(1)})^2 = 1$</span>

因此，<span class="math">$\alpha_1$</span> 完全由其他 <span class="math">$\alpha_i$</span> 决定，如果我们想固定 <span class="math">$\alpha_2, \ldots, \alpha_m$</span>，那么我们无法改变 <span class="math">$\alpha_1$</span> 而不违反优化问题中的约束（19）。

到这里，如果我们想更新某些 <span class="math">$\alpha_i$</span>，我们必须至少同时更新其中的两个以满足约束条件。于是，我们导出 SMO 算法，该算法只做以下操作：

<div class="math">
$$
\begin{aligned}
&\text{Repeat Until Convergence}\ \{ \\[5pt]
&\quad 1.\ 选择一对\ \alpha_i\ 和\ \alpha_j\ 进行更新\text{（使用启发式方法尝试选择两者，以使我们能取得最大进展朝向全局最大值）}\\[5pt]
&\quad 2.\ 重新优化\ W(\alpha)\ 关于\ \alpha_i\ 和\ \alpha_j，同时保持所有其他的\ \alpha_k\text{（}k \neq i, j\text{）固定 }\\[5pt]
&\}
\end{aligned}
$$
</div>

为了测试这个算法的收敛性，我们可以检查是否满足 KKT 条件（公式14-16），通常这些条件足以确定解的优化性。在这里，0.01 是收敛的公差参数，用于检测算法的停止条件（请参考其他有效的详细算法来理解为什么 SMO 是一个有效的迭代求解器）。

其关键原因是，更新 <span class="math">$\alpha_i, \alpha_j$</span> 时，SMO算法可以有效地选择主导的更新。下面是其核心思想。

假设我们想要固定 <span class="math">$\alpha_3, \ldots, \alpha_m$</span>，并希望重新优化 <span class="math">$\alpha_1$</span> 和 $\alpha_2$</span>（受约束条件（19）的限制）。由条件（19）我们得到：

<div class="math">
$$
\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = - \sum_{i=3}^m \alpha_i y^{(i)}
$$
</div>

由于右边的和是固定的（因为 <span class="math">$\alpha_3, \ldots, \alpha_m$</span> 是固定的），我们可以仅用一个常数 <span class="math">$\zeta$</span> 表示它：

<div class="math">
$$
\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta
$$
</div>

我们可以将 <span class="math">$\alpha_1$</span> 和 <span class="math">$\alpha_2$</span> 上的约束在几何上呈现如下：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/08/10/pASsAl8.png" data-lightbox="image-6" data-title="SMO">
  <img src="https://s21.ax1x.com/2024/08/10/pASsAl8.png" alt="SMO" style="width:100%;max-width:475px;cursor:pointer">
 </a>
</div>

从约束条件（18）我们知道，<span class="math">$\alpha_1$</span> 和 <span class="math">$\alpha_2$</span> 必须位于图中所示的 <span class="math">$[0, C] \times [0, C]$</span> 区间内。同时图中也绘制了直线 <span class="math">$\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = g$</span>，从这些约束条件我们知道 <span class="math">$L \leq \alpha_2 \leq H$</span>；否则，<span class="math">$\alpha_2$</span> 不能同时满足边界约束（box constraint）和直线约束。在这个例子中，<span class="math">$L = 0$</span>。但是依情况不同，通常会有某个下界 <span class="math">$L$</span> 和某个上界 <span class="math">$H$</span>，确保 <span class="math">$\alpha_1, \alpha_2$</span> 能够位于 <span class="math">$[0, C] \times [0, C]$</span> 的区间内。利用方程（20），我们还可以将 <span class="math">$\alpha_1$</span> 写作 <span class="math">$\alpha_2$</span> 的函数：

<div class="math">
$$
\alpha_1 = (\zeta - \alpha_2 y^{(2)}) y^{(1)}
$$
</div>

> [!NOTE]
> <span class="math">$y^{(1)} \in \\{-1, 1\\}$</span>，得 <span class="math">$(y^{(1)})^2 = 1$</span>。因此，目标函数 <span class="math">$W(\alpha)$</span> 可以写作 <span class="math">$ W(\alpha_1, \alpha_2, \ldots, \alpha_m) = W((\zeta - \alpha_2 y^{(2)}) y^{(1)}, \alpha_2, \ldots, \alpha_m) $</span>

将 <span class="math">$\alpha_3, \ldots, \alpha_m$</span> 视为常数，你应该能够验证这个目标函数在 <span class="math">$\alpha_2$</span> 上是一个二次函数形式 <span class="math">$a\alpha_2^2 + b\alpha_2 + c$</span>，其中 <span class="math">$a, b, c$</span> 是一些合适的常数。如果我们忽略“盒”约束（18）（或等价地，认为 <span class="math">$L \leq \alpha_2 \leq H$</span>），那么我们可以通过将其导数设置为零来轻松最大化这个二次函数。我们让 <span class="math">$\alpha_2^{new,\ unclipped}$</span> 表示得到的 <span class="math">$\alpha_2$</span> 的值。如果我们想要最大化 <span class="math">$W$</span> 关于 <span class="math">$\alpha_2$</span> 但受到边界约束，那么我们可以通过取 <span class="math">$\alpha_2^{new,\ unclipped}$</span> 并将其“剪裁”到区间内找到最优值，使得：

<div class="math">
$$
\alpha_{2}^{new} =
\begin{cases}
H & \text{if } \alpha_{2}^{new,\ unclipped} > H \\[5pt]
\alpha_{2}^{new,\ unclipped} & \text{if } L \leq \alpha_{2}^{new,\ unclipped} \leq H \\[5pt]
L & \text{if } \alpha_{2}^{new,\ unclipped} L
\end{cases}
$$
</div>
最后，找到 <span class="math">$\alpha_{2, \text{new}}$</span> 之后，我们可以使用方程（20）回去找到 <span class="math">$\alpha_1$</span> 的最优值。

还有一些更细节的部分相当简单：其中一个是选择下一个要更新的 <span class="math">$\alpha_i$</span> 的启发式方法；另一个是如何运行 SMO 算法。

至此，有关支持向量机的部分就结束了。
