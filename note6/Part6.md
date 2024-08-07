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

考虑逻辑回归，其中概率 $ p(y = 1|x; \theta) $ 由 $ h_\theta(x) = g(\theta^T x) $ 建模。

如果 $ h_\theta(x) \geq 0.5 $，或等效地，如果 $ \theta^T x \geq 0 $，我们就会在输入 $ x $ 上预测 1。考虑一个正样本（$ y = 1 $），$ \theta^T x $ 越大，$ h_\theta(x) = p(y = 1|x; w, b) $ 也越大，因此我们对标签是 1 的 “置信度” 也越高。

我们可以认为如果 $ \theta^T x \gg 0 $，我们的预测非常确定 $ y = 1 $。类似地，我们认为逻辑回归在 $ \theta^T x \ll 0 $ 时对 $ y = 0 $ 做出高置信度的预测。考虑到训练集，我们似乎找到了一个很好的拟合训练数据的模型：

**如果我们能找到使 $ \theta^T x^{(i)} \gg 0 $ 的 $ \theta $ 确信 $ y^{(i)} = 1 $，且 $ \theta^T x^{(i)} \ll 0 $ 确信 $ y^{(i)} = 0 $。**

为了更直观地感受 **边界（margin）**，样例如下图所示。

其中 $ × $ 表示正训练样本，$ \text{o} $ 表示负训练样本，决策边界是由方程 $ \theta^T x = 0 $ 给出的直线，也称为 <strong>分隔超平面（separating hyperplane）</strong>

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/07/08/pkfPHRP.png" data-lightbox="image-6" data-title="margin">
  <img src="https://s21.ax1x.com/2024/07/08/pkfPHRP.png" alt="margin" style="width:100%;max-width:375px;cursor:pointer">
 </a>
</div>

请注意，点 A 离决策边界很远。如果我们被要求在 A 处对 $ y $ 的值进行预测，似乎我们应该确信 $ y = 1 $。相反，点 C 非常靠近决策边界，虽然它位于我们会预测 $ y = 1 $ 的决策边界的一侧，但似乎只需对决策边界稍作改动，就很容易使我们的预测变为 $ y = 0 $。因此，我们对 A 的预测比对 C 的预测置信度更高，而点 B 位于这两种情况之间。

进一步地，如果一个点离分隔超平面很远，那么其可能在我们的预测中更置信。也就是说，训练样本需要离边界尽可能地远。这样会使得我们的模型更加有效。

### SVM 定义

为了易于后续学习 SVM，我们首先需要引入一些新的符号来讨论分类问题。

我们将考虑一个用于二元分类问题的线性分类器，其标签为 $ y $ 并且特征为 $ x $。从现在开始，我们将使用 $ y \in \\{-1, 1\\} $ 来表示类标签。此外，我们不再用向量 $ \theta $ 来描述我们的线性分类器，而是使用参数 $ w, b $。于是将分类器写为：

<div class="math">
$$
h_{w,b}(x) = g(w^T x + b)
$$
</div>

如果 $ z \geq 0 $，那么有 $ g(z) = 1 $；否则 $ g(z) = -1 $。

这种 $ w, b $ 符号让我们可以更明确地单独处理截距项 $ b $ 和其他参数。同时，也放弃了之前让 $ x_0 = 1 $ 作为输入特征向量中的一个额外项。在这里，$ b $ 取代了 $ \theta_0 $，而 $ w $ 取代了 $[ \theta_1 \ldots \theta_n ]^T$。

> [!WARNING]
> **根据我们上面定义的 $ g $，我们的分类器将直接预测 1 或 −1（参考感知机算法），而不是首先估计 $ y = 1 $ 的概率（这是逻辑回归所做的）。**

> [!TIP]
> 逻辑回归与支持向量机的定义区别<br>
> <table>
>     <thead>
>         <tr>
>             <th>特性</th>
>             <th>逻辑回归</th>
>             <th>支持向量机</th>
>         </tr>
>     </thead>
>     <tbody>
>         <tr>
>             <td><strong>标签和输出</strong></td>
>             <td>输出标签为 {0, 1}，输出为概率</td>
>             <td>输出标签为 {-1, 1}，直接预测类别</td>
>         </tr>
>         <tr>
>             <td><strong>决策函数</strong></td>
>             <td>使用 sigmoid 函数预测概率，连续</td>
>             <td>使用符号函数（感知机），离散</td>
>         </tr>
>     </tbody>
> </table>

### 边界

#### 函数边界

对于一个给定的训练样本 $ \\{x^{(i)},y^{(i)}\\} $，定义关于 $ (w,b) $ 关于样本的 **函数边界（functional margin）**为：

<div class="math">
$$
\hat{\gamma}^{(i)} = y^{(i)}(w^T x + b)
$$
</div>

对于支持向量机，其决策分数通常表现为 $ w^T x + b $：
- 当 $ w^T x + b = 0 $ 时，点恰好在决策边界上，即函数边界。
- 当 $ w^T x + b > 0 $ 时，模型预测样本属于正类（$ y = 1 $）。
- 当 $ w^T x + b < 0 $ 时，模型预测样本属于负类（$ y = -1 $）。

通过之前的学习，我们得出结论：要使结果置信度更高，那么就要求 $ \left| w^T x + b \right| $ 需要尽可能的大。

考虑两种情况，当 $ y = 1 $ 时，有 $ w^T x + b \gg 0 $；当 $ y = -1 $ 时，有 $ w^T x + b \ll 0 $。

但是针对于函数边界，存在一个特殊的问题。

如果我们用 $ 2w $ 和 $ 2b $ 替换 $w$ 和 $b$ （进行放缩），就会得到 $ g(w^T x + b) = g(2w^T x + 2b) $，这不会改变 $ h_{w,b}(x) $ 预测结果的正负。然而，用 $(2w, 2b)$ 替换 $(w, b)$ 却会使函数边界增加 2 倍。也就是说，通过任意的对于 $w$ 和 $b$ 的放缩，使得函数边界在衡量置信度上失去了意义（但实际的决策平面未改变）。为了解决这个问题，我们希望能够添加某种规范化条件，比如 $\|\|w\|\|_2 = 1$。也就是说，我们可能会用 $(\displaystyle{\frac{w}{\|\|w\|\|_2}}, \displaystyle{\frac{b}{\|\|w\|\|_2}})$ 替换 $(w, b)$，并相应地考虑 $(\displaystyle{\frac{w}{\|\|w\|\|_2}}, \displaystyle{\frac{b}{\|\|w\|\|_2}})$ 的函数边界。

> [!NOTE]
> **分隔平面 $ w^T x + b = 0 $ 描述的是一个通过原点的超平面，其方向由向量 $ w $ 确定，偏移由 $ b $ 确定。当 $ w $ 和 $ b $ 同时乘以同一个非零常数时，超平面的方向和偏移比例保持不变，因此超平面本身也保持不变。**

给定训练集 $ S = \\{(x^{(i)}, y^{(i)}) ; i = 1, \dots, m\\} $，我们还定义了 $ (w, b) $ 相对于 $ S $ 的函数边界为最小的那个个体训练样本的函数边界。通过 $ \hat{\gamma} $ 表示，写为：

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

决策边界对应于 $ (w, b) $ 如图所示，同时还标记了向量 $ \vec{w} $。注意 $ \vec{w} $ 与超平面是正交的。也就是说，$ \vec{w} $ 表示了样本到超平面的一个法向量。

考虑点 $ A $，它代表某个训练样本的输入 $ x^{(0)} $ 与标签 $ y^{(0)} = 1 $。其到决策边界的距离 $ \gamma^{(0)} $ 由线段 $AB$ 给出。

如何确定 $\gamma^{(0)} $ 的值呢？已知 $ \displaystyle{\frac{\vec{w}}{\|\vec{w}\|}} $ 是一个指向 $ \vec{w} $ 同一方向的单位向量。由点 $ A $ 表示 $ x^{(0)} $，得点 $ B $ 为 $ x^{(0)} - \gamma^{(0)} \cdot (\displaystyle{\frac{\vec{w}}{\|\vec{w}\|}}) $。

> [!NOTE]
> 由于点 $ B $ 在决策边界上，从 $ A $ 到 $ B $ 的向量（$ \vec{AB} $）必须与 $ \vec{w} $ 方向一致，并且是垂直于边界的。因此，$ \vec{AB} $ 可以表达为 $ \vec{w} $ 单位向量的某个标量倍数，即：
>
> <div class="math">
> $$ 
> \vec{AB} = -\gamma^{(0)} \cdot \left(\frac{\vec{w}}{\|\vec{w}\|}\right)
> $$
> </div>
> 
> 这里使用负号，因为我们从 $ A $ 沿 $ \vec{w} $ 的方向“移动”到边界，若 $ \vec{w} $ 从 $ A $ 指向超平面外，则移动方向为 $ \vec{w} $ 的反方向。
>点 $ B $ 的位置 $ x^{(B)} $ 由下式给出：
>
> <div class="math">
> $$ 
> x^{(B)} = x^{(0)} + \vec{AB} = x^{(0)} - \gamma^{(0)} \cdot \left(\frac{\vec{w}}{\|\vec{w}\|}\right)
> $$
> </div>

考虑一般情况，决策边界上的所有点都满足方程 $ w^T x + b = 0 $。因此，有：

<div class="math">
$$
w^T \left(x^{(i)} - \gamma^{(i)} \cdot \frac{w}{\|w\|}\right) + b = 0
$$
</div>

求解 $ \gamma^{(i)} $ 得：

<div class="math">
$$
\gamma^{(i)} = \frac{w^T x^{(i)} + b}{\|w\|} = \left(\frac{w}{\|w\|}\right)^T x^{(i)} + \frac{b}{\|w\|}
$$
</div>

这是针对 $ A $ 在图中所处 "正样本" 位置的情况计算得出。更普遍地，我们定义一个训练样本 $ \\{x^{(i)},y^{(i)}\\} $ 相对于 $ (w, b) $ 的 **几何边界（geometric margin）** 为：

<div class="math">
$$
\gamma^{(i)} = y^{(i)} \left( \left(\frac{w}{\|w\|}\right)^T x^{(i)} + \frac{b}{\|w\|} \right)
$$
</div>

注意，如果 $ \|w\| = 1 $，那么函数边界就等于几何边界。同时，几何边界具有放缩不变性。如果我们用 $ 2w $ 和 $ 2b $ 替换 $ w $ 和 $ b $，几何边界不会改变。因为经过前面的推导，几何边界实际上是样本点到超平面的欧几里得距离。

几何边界所具有放缩不变性会为 SVM 带来许多优势：

- **模型简化与正则化**：
调整权重向量 $ w $ 和偏置 $ b $ 的尺度不会影响模型的决策边界。这允许在模型训练过程中引入正则化约束，如限制 $ \|w\| $ 的大小，从而防止过拟合。
- **参数选择的灵活性**：
在部署模型到不同硬件或软件环境时，可能需要对模型参数进行缩放以满足特定的性能或资源使用要求，缩放不变性确保了这种调整不会影响模型的表现。

最后，给定一个训练集 $ S = \\{(x^{(0)},y^{(0)}); i = 1, \ldots, m\\} $，我们也定义了 $ (w, b) $ 相对于 $ S $ 的几何边界为个体训练样本几何边界的最小值：

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
> **当 $ || w || = 1 $ 时，函数边界与几何边界相等。**

### 拉格朗日对偶

#### 数学准备

##### 函数的凹凸性

###### 凸函数

一个定义在实数上的函数 $ f(x) $ 被称为 **凸函数（convex function）**，如果对于所有 $ x_1, x_2 $ 在其定义域内，并且对于所有 $ \lambda $  满足 $ 0 \leq \lambda \leq 1 $，都有：

<div class="math">
$$
f(\lambda x_1 + (1-\lambda) x_2) \leq \lambda f(x_1) + (1-\lambda) f(x_2)
$$
</div>

这个性质被称为函数的 **凸性（convex）** 。这意味着函数的任意两点之间的线段都位于函数图形的上方或与之相接。一个典型的凸函数例子是 $ f(x) = x^2 $。

###### 凹函数

与凸函数相对的是 **凹函数（concave function）**，定义几乎相同，只是不等式的方向相反。一个函数 $f(x) $ 是凹函数，如果对于所有 $ x_1, x_2 $ 在其定义域内，并且对于所有 $ \lambda $ 满足 $ 0 \leq \lambda \leq 1 $，都有：

<div class="math">
$$
f(\lambda x_1 + (1-\lambda) x_2) \geq \lambda f(x_1) + (1-\lambda) f(x_2)
$$
</div>

这表示函数的任意两点之间的线段都位于函数图形的下方或与之相接。一个常见的凹函数例子是 $ f(x) = \log(x) $，在其定义域 $ x > 0 $ 上。

> [!NOTE]
> <font size=4>**函数凹凸性的判断**</font>
> - **几何直观**：凸函数类似于一个向上开口的碗，而凹函数则像一个向下开口的碗。
> - **导数判定**：
>   - **凸函数**：如果一个函数的二阶导数（如果存在的话）是非负的（$ f''(x) \geq 0 $），那么这个函数是凸函数。
>   - **凹函数**：如果一个函数的二阶导数是非正的（$ f''(x) \leq 0 $），那么这个函数是凹函数。

##### 无约束的优化问题

优化问题顾名思义，也就是要找寻一个值来最小化或者最大化一个目标函数，最简单的形式如下：

<div class="math">
$$
\min_{x \in \mathbb{R}^n} f(x)
$$
</div>

这个式子表达的含义是以 $x$ 为自变量，找到 $f(x)$ 的最小值。如果想要求的最大值，可以把 $ f $ 进行一个转换，这样把 $\text{max}_x f(x)$ 等价转换为 $\text{min}_x -f(x)$ 。

注意，$x$ 并非一个标量数值，而是 $n$ 维实数空间上的一个向量。

在线性回归中，我们需要最小化代价函数。对于这样的问题，我们通常是求导寻找驻点，即找到 $x$ 使得 $\nabla f(x') = 0$。

当然我们也知道，只有当 $f$ 是凸函数的时候，这样才能得到全局最优解，否则会出现局部最优解的情况。下面的讨论都是假设函数为凸函数。

##### 带有等式约束的优化问题

一般地，我们进行优化函数总会带有一些限制条件，称为约束。带有等式约束的优化问题是一种更为复杂的优化任务，其中除了要最小化或最大化一个目标函数外，还需要满足一个或多个等式约束条件。

给定一个目标函数 $ f(x) $，其中 $ x \in \mathbb{R}^n $，以及一组等式约束 $ g_i(x) = 0 $，其中 $ i = 1, \ldots, m $，问题可以定义为：

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

假设有一个需要优化的目标函数 $ f(x) $，同时还有一个或多个约束条件 $ g_i(x) = 0 $。定义拉格朗日函数以结合约束：
<div class="math">
$$
L(x, \lambda) = f(x) + \sum_{i=1}^{m} \lambda_i g_i(x)
$$
</div>

其中，$ \lambda_i $ 是对应于第 $ i $ 个约束条件 $ g_i(x) $ 的拉格朗日乘子。

为了找到目标函数 $ f(x) $ 在约束条件 $ g_i(x) = 0 $ 下的极值点，我们需要求解拉格朗日函数 $ L(x, \lambda) $ 对变量 $ x $ 和乘子 $ \lambda $ 的偏导数：
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

通过求解方程组，可以得到 $ x $ 和 $ \lambda $ 的值，这些值给出了目标函数在满足约束条件下的极值点。

> [!TIP]
> <font size=4>**拉格朗日乘数法中的参数解释**</font><br>
> **拉格朗日乘子** $\lambda $：它是用来引入约束条件 $ g_i(x) = 0 $ 到目标函数 $ f(x) $ 中的系数。每个约束条件 $ g_i(x) $ 对应一个拉格朗日乘子 $ \lambda_i $，通常用来确定约束条件对目标函数的影响。<br>
> **驻点**：通过拉格朗日函数的偏导数为零的点（驻点），可以找到目标函数在约束条件下的可能极值点。这些驻点满足了同时优化目标函数 $ f(x) $ 和约束条件 $ g_i(x) = 0 $ 的要求。

##### 仿射函数

一个函数 $ f: \mathbb{R}^n \to \mathbb{R}^m $ 被称为 **仿射函数（affine function）**，如果它可以表示为：

<div class="math">
$$
f(x) = A x + b
$$
</div>

其中：
- $ A $ 是一个 $ m \times n $ 的常数矩阵，
- $ b $ 是一个 $ m $ 维常向量，
- $ x $ 是一个 $ n $ 维向量。

例如：

一个从 $ \mathbb{R}^n $ 到 $ \mathbb{R}^1 $ 的典型仿射函数可以写作：

<div class="math">
$$
f(x) = a^T x + b
$$
</div>

其中：
- $ x $ 是一个 $ n $ 维向量，即 $ x = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^n $。
- $ a = (a_1, a_2, \ldots, a_n) $ 是一个 $ n $ 维常向量，即 $ a \in \mathbb{R}^n $。
- $ b $ 是一个常数，即 $ b \in \mathbb{R} $。

仿射函数具有如下一个重要的特性。对于任意 $ x, y \in \mathbb{R}^n $ 和任意 $ \lambda \in [0, 1] $：

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
- $ f(x) $ 是凸函数，称为目标函数。
- $ g_i(x) $ 是凸函数，表示不等式约束。
- $ h_j(x) $ 是仿射函数，表示等式约束。
- $ \mathbb{R}^n $ 中的 $ x $ 是优化变量。

凸优化问题具有一个重要的性质：**任何凸优化问题都具有全局最优解**。

##### 对偶问题

###### 线性规划

在进一步了解对偶问题前，我们需要先了解 **线性规划（linear programming）**。

一个典型的线性规划问题可以表示为：

- **目标函数**：需要进行最大化或最小化的线性函数。

<div class="math">
$$
\text{maximize} \quad c^T x \quad \text{or} \quad \text{minimize} \quad c^T x
$$
</div>

其中 $ c $ 和 $ x $ 均为向量。

- **约束条件**：这些是问题的限制条件，形式为线性等式或不等式。

<div class="math">
$$
Ax \leq b, \quad Ax = b, \quad \text{or} \quad Ax \geq b
$$
</div>

其中 $ A $ 是一个矩阵，$ b $ 是一个向量。

- **变量的非负限制**：在大多数线性规划问题中，决策变量需要非负。

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

工厂除了拿原料生产成产品卖掉这条出路外，还有一种方法是直接将原料卖掉。但是这种情况只存在于把原料卖掉赚的钱比生产成产品赚的钱多，才去会这样做。那么最低可以接受多少的价格呢？假设原料A和B的单价分别为：$w_1$ 和 $w_2$，那么可以用数学表达式描述如下：

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

这里，$\alpha_i$ 和 $\beta_i$ 是拉格朗日乘子。考虑以下量：

<div class="math">
$$
\theta_\mathcal{P}(w) = \max_{\alpha, \beta:\alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta)
$$
</div>

这里，$ \mathcal{P} $ 代表 "原问题"。让我们假设一个特定的 $w$ 满足原问题的约束条件（即，如果 $g_i(w) > 0$ 或 $h_i(w) \neq 0$ 对某些 $ i $）。那么有：

<div class="math">
$$
\begin{align*}
\theta_\mathcal{P}(w) &= \max_{\alpha, \beta; \alpha_i \geq 0} f(w) + \sum_{i=1}^{k} \alpha_i g_i(w) + \sum_{i=1}^{l} \beta_i h_i(w) \tag{1} \\[5pt]
&= \infty \tag{2}
\end{align*}
$$
</div>

相反，如果一个特定的 $w$ 满足约束条件，那么 $\theta_\mathcal{P}(w) = f(w)$。就有：

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



因此，$\theta_\mathcal{P}(w)$ 在我们的问题中对于所有满足原始约束的 $w$ 值都是相同的，并且如果约束条件被违反，则会是正无穷。所以，我们考虑下面的最小化问题：

<div class="math">
$$
\min_{w} \ \theta_\mathcal{P}(w) = \min_{w} \ \max_{\alpha, \beta} \mathcal{L}(w, \alpha, \beta)
$$
</div>

我们看到这是相同的问题，即与原始问题一样，并且有相同的解，我们将原始问题的最优值定义为 $p^* = \text{min} \ \theta_\mathcal{P}(w)$；这是原始问题的目标值。

现在，让我们看一个稍微不同的问题。我们定义：

<div class="math">
$$
\theta_\mathcal{D}(\alpha, \beta) = \min_{w} \mathcal{L}(w, \alpha, \beta)
$$
</div>

这里，$ \mathcal{D} $ 代表 "对偶"。注意在 $\theta_\mathcal{D}$ 的定义中我们是在最小化关于 $ w $ 而最大化关于 $\alpha, \beta$，这里我们是在最小化关于这些参数的问题：

<div class="math">
$$
\max_{\alpha, \beta:\alpha_i \geq 0} \ \theta_\mathcal{D}(\alpha, \beta) = \max_{\alpha, \beta:\alpha_i \geq 0} \ \min_{w} \mathcal{L}(w, \alpha, \beta)
$$
</div>

这完全是同样的问题，只是 "max" 和 "min "的顺序被交换了。我们定义对偶问题的最优值为 $d^* = \max_{\alpha,\beta}\ \theta_\mathcal{D}(\alpha, \beta)$。

如何证明原问题和对偶问题之间的关系？可以容易地显示出：

<div class="math">
$$
d^* = \max_{\alpha, \beta; \alpha_i \geq 0} \min_w \mathcal{L}(w, \alpha, \beta) \leq \min_w \max_{\alpha, \beta; \alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta) = p^*
$$
</div>

在某些约束条件下，我们会有 $p^* = d^*$。

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

即便原问题不是凸问题，上述不等式也成立，这就是 **弱对偶性（Weak Duality）**。这种性质即使是 $d^*, p^*$为 无穷也成立:

- 如果 $p^* = -\infty$，则 $d^* = -\infty$
- 如果 $d^* = +\infty$，则 $p^* = +\infty$

我们还定义 $p^* - d^*$ 称为 **最优对偶间隙（Optimal Duality Gap）**，这个值必然非负。

##### 强对偶

当 $ d^* = p^*$ 时，称为 **强对偶性（Strong Duality）**。

下面是强对偶的示意图：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/07/15/pk5TPSJ.png" data-lightbox="image-6" data-title="strong duality">
  <img src="https://s21.ax1x.com/2024/07/15/pk5TPSJ.png" alt="strong duality" style="width:100%;max-width:500px;cursor:pointer">
 </a>
</div>

显然，强对偶需要满足一定的限制条件。

#### KKT 条件

在之前的假设下，必须存在 $w^*$, $\alpha^*$, $\beta^*$ 使得 $w^*$ 是原问题的解，$\alpha^*$, $\beta^*$ 是对偶问题的解。此外，$p^* = d^* = \mathcal{L}(w^*, \alpha^*, \beta^*)$。同时，$w^*$, $\alpha^*$ 和 $\beta^*$ 满足  **Karush-Kuhn-Tucker（KKT）条件**，具体如下：

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

如果某个 $w^*$, $\alpha^*$, $\beta^*$ 满足 KKT 条件，则它们同时也是原问题和对偶问题的解。

我们特别指出公式（5），这是所谓的 KKT 对偶互补条件。具体来说，如果 $\alpha_i^* > 0$，则 $g_i(w^*) = 0$。（即，$g_i(w) \leq 0$ 约束是活跃的，意味着它以等式而不是不等式的形式成立。）后续，这将是证明 SVM 中仅有少数 "支持向量" 的关键；KKT 对偶互补条件也将在我们讨论 SMO 算法时，为我们提供收敛性测试。

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

在上述条件中，通过 $ ||w|| = 1 $ 使得函数边界等于几何边界，从而将两者联系起来。同时，保证了几何边界的最小值是 $\gamma$。解决上述问题将得到 $ (w, b) $ 相对于训练集 $ S $ 的最大几何边界。

但是，$ ||w|| = 1 $ 是一个非凸约束，不容易直接解出上述问题。所以，考虑将问题转化为另一个形式：

<div class="math">
$$
\begin{aligned}
\max&_{\hat{\gamma},w,b}\ \ \frac{\hat{\gamma} }{||w||}\\[5pt]
\text{s.t.}& \quad \quad \ y^{(i)}(w^T x^{(i)}+b) \geq \hat{\gamma},\quad i = 1,2,...,m
\end{aligned}
$$
</div>

很不幸，$\displaystyle{\frac{\hat{\gamma} }{||w||}}$ 仍然是非凸的。继续将问题进行转化。

在之前的对于边界的讨论中，我们提到函数边界可以随 $ (w, b) $ 进行任意的放缩。于是，在这里我们规定，$ (w, b) $ 相对于 $ S $ 的函数边界为 1 ：

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

显然，$\displaystyle{\frac{1}{2}||w||^2}$ 是一个带有凸二次目标函数和线性约束的优化问题。它的解给我们提供了最优边界分决策器。这种优化问题可以使用常见的 **二次规划（quadratic programming）** 算法来解决。

在基本了解了最优边界决策器后，我们将通过我们之前了解到的拉格朗日对偶来进一步深入。

将约束转化为：

<div class="math">
$$
g_i(w) = -y^{(i)}(w^T x^{(i)}+b) + 1 \leq 0
$$
</div>

由 KKT 条件的等式$(5)$与等式$(7)$得：

拉格朗日算子 $ \alpha_i \geq 0 $ 。若要使 $ \alpha_i \cdot g_i(w) = 0 $，则有 $ \alpha_i = 0 $ 或 $ g_i(w) = 0 $。

当 $ \alpha_i > 0 $ 时，且仅当 $ g_i(w) = 0 $ 满足条件。即 $ -y^{(i)}(w^T x^{(i)}+b) + 1 = 0$。自然得到：$y^{(i)}(w^T x^{(i)}+b) = 1$，即 $ \hat{\gamma} = 1$。满足上述条件的样本点，一定是距离决策边界最近的，如下图所示：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/07/15/pk5H1dP.png" data-lightbox="image-6" data-title="support vectors">
  <img src="https://s21.ax1x.com/2024/07/15/pk5H1dP.png" alt="support vectors" style="width:100%;max-width:400px;cursor:pointer">
 </a>
</div>

这些在虚线上，距离决策边界最近的点被称为 **支持向量（support vector）**。支持向量的特性是函数边界 $ \hat{\gamma} = 1$ 且其拉格朗日算子 $ \alpha_i > 0 $。

当 $ \alpha_i = 0 $，此时单独考虑 $ g_i(w) = -y^{(i)}(w^T x^{(i)}+b) + 1 < 0 $。对其进行等价转换得 $ g_i(w) = y^{(i)}(w^T x^{(i)}+b) > 1 $。表示这些点未越过由支持向量构成得决策平面，它们被正确地分类。

接下来我们对上述优化问题的对偶问题进行讨论。

当我们尝试利用拉格朗日对偶来解决此问题时，其中一个关键的点在于只用输入特征的内积  $ \langle x^{(i)}, x^{(j)} \rangle $ 来表示算法。

> [!NOTE]
> **$ \langle x^{(i)}, x^{(j)} \rangle $** 等价于 **$ (x^{(i)})^T x^{(j)} $。**

当我们为我们的对偶优化问题构建拉格朗日函数时，我们有：

<div class="math">
$$
\begin{align*}
\mathcal{L}(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^m \alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1] \tag{8}
\end{align*}
$$
</div>
注意这里只有 $\alpha_i$，而没有 $\beta_i$。表明拉格朗日乘子只有不等式约束。

接下来，需要找到问题的对偶形式。为此，我们首先需要对 $w$ 和 $b$（对于固定的 $\alpha$）求导并令其等于零以最小化 $\mathcal{L}(w, b, \alpha)$，进而得到 $θ_\mathcal{D}$。于是，我们有：

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

其对于 $b$ 的导数为：

<div class="math">
$$
\frac{\partial}{\partial b} L(w, b, \alpha) = \sum_{i=1}^m \alpha_i y^{(i)} = 0 \tag{10}
$$
</div>

如果取 $w$ 在方程$(9)$中的定义并将其重新代入拉格朗日方程（方程$(8)$），并化简，会得到：

<div class="math">
$$
L(w, b, \alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j x^{(i)T} x^{(j)} - b \sum_{i=1}^m \alpha_i y^{(i)}
$$
</div>

又根据方程（10），得到最后一项必须为零，因此有：

<div class="math">
$$
L(w, b, \alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j (x^{(i)})^T x^{(j)}
$$
</div>

由于我们是通过最小化 $ \mathcal{L}$ 相对于 $w$ 和 $b$ 的函数得到上述方程的，并且一直有 $\alpha_i \geq 0 $ 的约束和方程$(10)$。于是，我们得到以下对偶优化问题：

<div class="math">
$$
\begin{aligned}
\text{max}_\alpha \quad &W(\alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle \\[5pt]
\text{s.t.}\quad &\alpha_i ≥ 0, i = 1, \ldots, m \\[5pt]
&\sum_{i=1}^m \alpha_i y^{(i)} = 0
\end{aligned}
$$
</div>

您也应该能够验证 $p^* = d^*$ 和 KKT 条件（方程 3-7）确实适用于我们的优化问题。因此，我们可以解决对偶问题而不是原始问题。具体来说，在上述的对偶问题中，我们有一个最大化问题，其参数是 $\alpha_i$。我们稍后再谈这个问题。

关于我们将用于解决对偶问题的特定算法的讨论可以留到后面，但如果我们确实能解决它（即，找到最大化 $W(\alpha)$ 的 $\alpha$ 值，并满足约束条件，那么我们可以使用方程$(9)$回去找到 $\alpha$ 的函数作为最优 $w$ 的解。找到 $w^*$ 后，通过考虑原始问题，找到截距项 $b$ 的最优值也很直接。具体而言，截距项 $b$ 可以表示为：

<div class="math">
$$
b^* = -\frac{\max_{y^{(i)}=1} w^{*T}x^{(i)} + \min_{y^{(i)}=-1} w^{*T}x^{(i)}}{2} \tag{11}
$$
</div>

在继续之前，让我们也仔细看看方程$(9)$，它给出了最优 $w$ 的值，即 $\alpha$ 的最优值的函数。假设我们已经将我们模型的参数适配到一个训练集上，现在希望对一个新的输入点 $x$ 进行预测。我们将计算 $w^T x + b$ 并且只有当这个量大于零时，才预测 $y = 1$。但是使用方程$(9)$，这个量也可以写成：

<div class="math">
$$
\begin{align*}
w^T x + b = \left( \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)} \right)^T x + b \tag{12} \\[5pt]
= \sum_{i=1}^m \alpha_i y^{(i)} \left( x^{(i)} \right)^T x + b \tag{13}
\end{align*}
$$
</div>

因此，如果我们找到了 $\alpha_i$，为了做出预测，我们需要计算一个只依赖于 $x$ 和所有训练集数据点之间内积的量。我们之前已经看到 $\alpha_i$ 将全部为零，除了支持向量。因此，许多在总和上述的项将为零，而我们真正需要考虑的只是内积在 $x$ 和支持向量之间的（通常只有少数几个）来计算$(13)$并做出我们的预测。

通过检查问题的对偶形式，我们获得了问题结构的重要见解，并且还能够完全用内积项将整个算法表达出来，仅应用到特征向量。在下一节中，我们将利用这个属性应用内核技巧到我们的分类问题中。结果算法，支持向量机，将能够在非常高维的空间中有效学习。

### 核 Kernel

在之前关于线性回归的讨论中，我们遇到了一个问题，其中输入 $ x $ 代表了房屋的居住面积，我们考虑执行回归分析。使用特征 $ x, x^2 $ 和 $ x^3 $ 来获得一个三次函数。为了区分这两组变量，我们将原始输入值称为问题的输入属性（在这个例子中，$ x $，即居住面积）。当这些属性映射到一些新的量，然后传递给学习算法时，我们将这些新的量称为输入特征。不同的作者使用不同的术语来描述这两种事物，但我们会尝试在这些笔记中一致使用这些术语。我们还将让 $ \phi $ 表示特征映射，它将属性映射到特征。例如，在我们的例子中，我们有：

<div class="math">
$$
\phi(x) = \begin{bmatrix} x \\ x^2 \\ x^3 \end{bmatrix}
$$
</div>

而不是使用 SVM 的原始输入属性 $ x $ 应用学习算法，我们可能想要使用一些特征 $ \phi(x) $ 来学习。为此，我们只需回顾之前的算法，并用 $ \phi(x) $ 替换其中的 $ x $。因为算法完全可以用内积的形式编写，这意味着我们将用 $ \phi(x), \phi(z) $ 替换所有这些内积。具体来说，给定一个特征映射 $ \phi $，我们定义相应的核为：

<div class="math">
$$
K(x, z) = \phi(x)^\top \phi(z)
$$
</div>

然后，在我们之前算法中使用的任何地方都用 $ K(x, z) $ 替换 $ x, z $，现在我们的算法将使用特征 $ \phi $ 学习。现在，给定 $ \phi $，我们可以通过找到 $ \phi(x) $ 和 $ \phi(z) $ 并取它们的内积来轻松计算 $ K(x, z) $。但更有趣的是，尽管 $ \phi(x) $ 本身可能计算成本很高，$ K(x, z) $ 的计算通常很便宜。在这种情况下，通过在算法中使用一个有效的方式来计算 $ K(x, z) $，我们可以让 SVM 在由 $ \phi $ 给定的高维特征空间中学习，而不需要显式地找到或表示向量 $ \phi(x) $。

这是一个例子。假设 $ x, z \in \mathbb{R}^n $，并考虑

<div class="math">
$$
K(x, z) = (x^\top z)^2
$$
</div>

我们还可以将其写成以下形式：

<div class="math">
$$
K(x, z) = \left(\sum_{i=1}^n x_iz_i\right)^2 = \sum_{i=1}^n \sum_{j=1}^n x_iz_ix_jz_j = \sum_{i,j=1}^n (x_ix_j)(z_iz_j)
$$
</div>

因此，我们看到 $ K(x, z) $ 等于 $ \phi(x)^\top \phi(z) $，其中特征映射 $ \phi $ 给出如下（这里展示的是 $ n = 3 $ 的情况）：

<div class="math">
$$
\phi(x) = \begin{bmatrix}
x_1x_1 \\
x_1x_2 \\
x_1x_3 \\
x_2x_1 \\
x_2x_2 \\
x_2x_3 \\
x_3x_1 \\
x_3x_2 \\
x_3x_3
\end{bmatrix}
$$
</div>

请注意，尽管计算高维度的 $ \phi(x) $ 需要 $ O(n^2) $ 时间，但找到 $ K(x, z) $ 只需要 $ O(n) $ 时间——线性于输入属性的维度。

考虑相关的核，请也考虑以下形式：

<div class="math">
$$
K(x, z) = (x^\top z + c)^2 = \sum_{i=1}^n (x_iz_i)(z_ix_i) + \sum_{i=1}^n \sqrt{2cx_i} \sqrt{2cz_i} + c^2
$$
</div>

（请自行验证。）这对应于特征映射（再次展示如下）：

<div class="math">
$$
\phi(x) = \begin{bmatrix}
x_1x_1 \\
x_1x_2 \\
\vdots \\
x_nx_n \\
\sqrt{2cx_1} \\
\vdots \\
\sqrt{2cx_n} \\
c
\end{bmatrix}
$$
</div>

参数 $ c $ 控制了 $ x_i $（一阶项）和 $ x_ix_j $（二阶项）之间的相对权重。更广泛地说，核 $ K(x, z) = (x^\top z + c)^d $ 对应于一个特征映射到一个 $ n^d $ 维特征空间，涵盖了所有的 $ x_i, x_i \cdot x_j $ 等到 $ d $ 阶的单项式。然而，尽管在这个 $ n^d $-维空间中操作，计算 $ K(x, z) $ 仍然只需要 $ O(n^2) $ 时间，因为我们不需要显式地表示这个非常高维的特征向量。

现在，让我们从一个稍微不同的角度来考虑核的概念。直观上，如果 $ \phi(x) $ 和 $ \phi(z) $ 非常接近，则 $ K(x, z) = \phi(x)^\top \phi(z) $ 应该很大。相反，如果 $ \phi(x) $ 和 $ \phi(z) $ 非常远离——比如几乎正交——那么 $ K(x, z) $ 将会很小。因此，我们可以将 $ K(x, z) $ 看作是 $ \phi(x) $ 和 $ \phi(z) $ 的相似度或者接近度的量度。

给定这种直觉，假设你正在处理某个学习问题，并且你想到了一个你认为可能是 $ x $ 和 $ z $ 相似度的合理度量的函数 $ K(x, z) $，比如你选择：

<div class="math">
$$
K(x, z) = \exp\left(-\frac{\|x-z\|^2}{2\sigma^2}\right)
$$
</div>

这是一个衡量 $ x $ 和 $ z $ 相似度的合理度量，当 $ x $ 和 $ z $ 接近时接近 1，当 $ x $ 和 $ z $ 远离时接近 0。我们可以使用这个定义作为核，答案是肯定的。这个核被称为高斯核，在某些文献中，也被称为径向基函数（RBF）核。

要到达一个无限维的特征映射 $ \phi $。但更广泛地说，给定某些函数 $ K $，我们如何确定它是否是一个有效的核，即我们是否可以找到某种特征映射 $ \phi $ 使得 $ K(x, z) = \phi(x)^\top \phi(z) $ 对所有的 $ x, z $ 成立？假设现在 $ K $ 确实是一个有效的核，对应某个特征映射 $ \phi $。现在，考虑一组有限的 $ m $ 个点的集合（不一定是训练集）{ $ x_1, ..., x_m $ }，并且定义一个 $ m \times m $ 的矩阵 $ K $ ，其 $ (i, j) $-项由 $ K_{ij} = K(x_i, x_j) $ 给出。这个矩阵被称为核矩阵。注意，我们重载了符号 $ K $ 来表示核函数 $ K(x, z) $ 和核矩阵 $ K $，它们显然是密切相关的。

如果 $ K $ 是一个有效的核，那么 $ K_{ij} = K(x_i, x_j) = \phi(x_i)^\top \phi(x_j) = \phi(x_j)^\top \phi(x_i) = K_{ji} $，因此 $ K $ 必须是对称的。此外，如果让 $ \phi_k(x) $ 表示向量 $ \phi(x) $ 的第 $ k $ 个坐标，我们发现对于任何向量 $ z $，我们有：

<div class="math">
$$
z^\top Kz = \sum_i \sum_j z_iK_{ij}z_j = \sum_i \sum_j z_i\phi(x_i)^\top \phi(x_j)z_j = \sum_k (\sum_i z_i\phi_k(x_i))^2 \geq 0
$$
</div>

上述倒数第二步使用了之前问题集中的同一个技巧。由于 $ z $ 是任意的，这表明 $ K $ 是半正定的。

因此，如果 $ K $ 是一个有效的核（即它对应于某个特征映射 $ \phi $），那么相应的核矩阵 $ K $ 是对称正定的。更一般地说，这不仅是 $ K $ 是有效核的必要条件，也是充分条件（也称为 Mercer 核）。Mercer 定理给出了这一结果，对于 $ K $ 是有效的 Mercer 核，它是必要且充分的条件，即对于任何 { $ x_1, ..., x_m $ }，$ m < \infty $，相应的核矩阵 $ K $ 是对称正定的。

### 正则化

### SMO 算法
