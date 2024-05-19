# 第二章&ensp;线性回归

引入以下符号：

* $x\_j^{(i)}$：第 $i$ 个训练样本的特征 $j$ 的值
* $x^{(i)}$：第 $i$ 个训练样本的所有特征
* $m$：训练样本的数量
* $n$：特征的数量

### 假设函数

对于一个多元的线性回归问题，我们假设其有 $ n $ 个特征，有这样一个 **假设函数 (hypothesis function)** ：
<div class="math">
$$
h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n \ \ \ (其中\ x_0 = 1)
$$
</div>

转化为矩阵形式有：
<div class="math">
$$
h_\theta(x) = \theta^Tx
$$
</div>

即：
<div class="math">
$$
h\_{\theta}(x) = \big[\theta\_0,\ \theta\_1,\ ... ,\ \theta\_n\big] \left[\begin{matrix}x\_0\\\ x\_1\\\ ...\\\ x\_n\end{matrix}\right]= \theta^Tx
$$
</div>

其中，<sapn class="math">$\theta$</span>  表示  <span class="math">$\theta_0, \theta_1, \theta_2, \cdots, \theta_n$</span> 组成的列向量。

### 代价函数

在线性回归中，一般采用一个标准来衡量模型训练的好坏，即 **代价函数 (cost function)** ：

<div class="math">
$$
J(\theta_0, \theta_1, \theta_2, \cdots, \theta_n) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2
$$
</div>

由代价函数的定义我们可以得知：当 $h_\theta(x)$ 越接近 $y$ 时，代价函数的值越小，即模型训练的效果越好。



### LMS Algorithm 

为了最小化代价函数以得到最优的模型，我们采用 **最速梯度下降 (Greadient Descent)** 算法来更新参数 $ \theta $：

<div class="math">
$$
\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta)
$$
</div>

* 注意：这里的参数 $ \theta = \big[\theta\_0,\ \theta\_1,\ \theta\_2, \cdots, \theta\_n\big] $ 是 **同时更新** 的。

其中，$\alpha$ 表示 **学习率 (learning rate)** ，其值决定了梯度下降的步长，通常取一个较小的值。

下面对于函数求导部分进行推导，对于其中一个训练样本：

<div class="math">
$$
\begin{align}
\frac{\partial}{\partial\theta_j}J(\theta) &= \frac{\partial}{\partial\theta_j}\frac{1}{2}(h_\theta(x) - y)^2 \\[5pt]
&= 2 \cdot \frac{1}{2}(h_\theta(x) - y) \cdot \frac{\partial}{\partial\theta_j}(h_\theta(x) - y) \\[5pt]
&= (h_\theta(x) - y) \cdot \frac{\partial}{\partial\theta_j}(\sum\limits_{i=0}^{n}\theta_ix_i - y) \\[5pt]
&= (h_\theta(x) - y)x_j
\end{align}
$$
</div>

如此，对于一个训练样本，其更新规则为：

<div class="math">
$$
\theta_j := \theta_j + \alpha(y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}
$$
</div>

这就是我们所说的 **LMS Update Rule**（LMS表示 "least mean squares"），即 **最小均方算法** 。

借由 LMS，我们得到了两种学习方法：

* 批量梯度下降 (Batch Gradient Descent)
* 随机梯度下降 (Stochastic Gradient Descent)

#### 批量梯度下降

批量梯度下降 (Batch Gradient Descent) 指的是每次更新参数时，使用全部的训练样本。

<div class="math">
$$
\theta_j := \theta_j + \alpha\frac{1}{m}\sum\limits_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
$$
</div>

其算法描述为：

<div class="math">
$
\begin{align}
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ &\mathrm{Repeat\ Until\ Convergence}\ \{ \\[5pt]
&\ \ \ \ \ \ \ \theta_j := \theta_j + \alpha\sum\nolimits_{i=1}^{m}(y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}\ \ \ \ \ \ \ (\mathrm{for\ every\ } j). \\[5pt]
&\}
\end{align}
$
</div>

#### 随机梯度下降

随机梯度下降 (Stochastic Gradient Descent) 指的是每次更新参数时，使用一个随机的训练样本。

其算法描述为：

<div class="math">
$
\begin{align}
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ &\mathrm{Loop}\ \{ \\[5pt]
&\ \ \ \ \ \ \mathrm{for}\ i = 1\ \mathrm{to}\ m, \{ \\[5pt]
&\ \ \ \ \ \ \ \ \ \ \ \ \theta_j := \theta_j + \alpha(y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}\ \ \ \ \ \ \ (\mathrm{for\ every\ } j). \\
&\ \ \ \ \ \ \} \\
&\}
\end{align}
$
</div>

### ⭐小结：LMS

#### BGD 与 SGD 的区别
批量梯度下降法（BGD）和随机梯度下降法（SGD）是都采用了LMS的核心思想，但其区别在于：

* 批量梯度下降法每次更新参数时，使用全部的训练样本。
* 随机梯度下降法每次更新参数时，使用一个随机的训练样本。

如果训练集过大，一般采用 SGD 以加速收敛。

#### SGD 与 学习率衰减

SGD在训练过程中虽然收敛速度更快，但是存在一个问题：SGD最后会最优处来回振荡。这导致其无法收敛到最优解。所以在训练过程中我们一般采用如下策略来应对：

> <div class="blocks">在学习过程中，逐渐减小学习率至0，以保证在训练过程中能够收敛到最优解，而非来回振荡。</div>

这种策略被称为 **学习率衰减 (learning rate decay)** 。

虽然但是，往往在实际的训练中，来回振荡的值已经逼近了最优解，所以也可以近似地将这个值视作最优解来使用。

### Normal Equation

除开梯度下降之外，还有另一种最小化代价函数的方法： **正规方程 (normal equation)** 。

与梯度下降不同的是：正规方程明确的推导出了最小化的结果，而非像梯度下降一般需要进行迭代演算。

#### 数学准备

##### 矩阵求导入门

对于一个将 $ m×n $ 的矩阵映射到实数的函数 $ f $，定义 $ f:\mathbb{R}^{m×n} \mapsto \mathbb{R}$ 对于 $ A $ 的导数为：

<div class="math">
$$
\nabla_A f(A) = \begin{bmatrix}
   \frac{\partial f}{\partial A_{11}} & \ldots & \frac{\partial f}{\partial A_{1n}} \\
   \vdots & \ddots & \vdots \\
   \frac{\partial f}{\partial A_{m1}} & \ldots &  \frac{\partial f}{\partial A_{mn}}
 \end{bmatrix}
$$
</div>

也就是说，函数对于矩阵求导就是函数对矩阵里每一项按顺序求导，得到的结果是一个与 $ A $ 相同大小的矩阵。

下面是一个简单的例子，假设我们有一个矩阵 $ A $，其大小为 $ 2×2 $，其元素为：

<div class="math">
$$
A = \begin{bmatrix}
   A_{11} & A_{12} \\
   A_{21} & A_{22}
 \end{bmatrix}
$$
</div>

那么，对于函数 $ f: $

<div class="math">
$$
f(A) = \frac{3}{2}A_{11} + 5A^2_{12} + A_{21}A_{22}
$$
</div>

其对于矩阵 $ A $ 的导数为：

<div class="math">
$$
\nabla_A f(A) = 
    \begin{bmatrix}
        \frac{3}{2} & 10A_{12} \\
        A_{22} & A_{21} \\
    \end{bmatrix}
$$
</div>

##### 矩阵的迹

对于一个 $ n×n $ 的矩阵 $ A (方阵) $ ，其 **迹 (trace)** 定义为：

<div class="math">
$$
\mathrm{tr}(A) = \sum\limits_{i=1}^{n}A_{ii}
$$
</div>

也就是说，矩阵的迹等于矩阵的主对角线的元素之和。

##### ⭐迹的性质

以下我们有一些容易证明的有关迹的性质，当 $ A $ 和 $ B $ 都是方阵时：

<div class="math">
$$
\mathrm{tr}AB = \mathrm{tr}BA
$$
</div>

<div class="math">
$$
 \mathrm{tr}(A+B) = \mathrm{tr}A + \mathrm{tr}B
$$
</div>

<div class="math">
$$
\mathrm{tr}A = \mathrm{tr}A^T
$$
</div>

<div class="math">
$$
\mathrm{tr}aA = a\mathrm{tr}A
$$
</div>

然后我们有迹的 **循环不变性**：

<div class="math">
$$
\mathrm{tr}ABC = \mathrm{tr}CAB = \mathrm{tr}BCA
$$
</div>

进一步地，我们列出一些后面推导中会用到的性质：

<div class="math">
$$
\begin{align}
&\nabla_{A} \text{tr} AB = B^T& \tag{1}\\ \\
&\nabla_{A}^T f(A) = (\nabla_{A} f(A))^T& \tag{2}\\ \\
&\nabla_{A}\text{tr}AB A^TC = CAB + C^TA B^T& \tag{3}
\end{align}
$$
</div>

#### ⭐方程推导

接下来，我们将利用之前提到的性质对方程进行数学推导。

将每一个参数 $ \theta $ 对应的 $ x $ 值看做是一个列向量，那么特征矩阵 $ X $ 可以表示为：

<div class="math">
$$
X = \begin{bmatrix}
    - & (x^{(1)})^T & - \\
    - & (x^{(2)})^T & - \\
    & \vdots & \\
    - & (x^{(m)})^T & - \\
\end{bmatrix}
\begin{aligned}
\quad \text{(size: m × (n+1))} 
\end{aligned}
$$
</div>

其中，$ X $ 的大小为 $ m × (n+1) $ 。这里还是假设了 $ x_0 = 1 $，所以 $ X $ 的第一列全为 $ 1 $ 。

同样的，我们把 $ y $ 也写成一个列向量形式：

<div class="math">
$$
\vec{y} = \begin{bmatrix}
    y^{(1)} \\
    y^{(2)} \\
    \vdots \\
    y^{(m)}
\end{bmatrix}
\begin{aligned}
\quad \text{(size: m × 1)} 
\end{aligned}
$$
</div>

之前我们提到过假设函数的矩阵表示形式：$h_\theta(x^{(i)}) = (x^{(i)})^T \theta$，那么进行等价代换有：

<div class="math">
$$
\begin{aligned}
X \theta - \vec{y} &= \begin{bmatrix}
    (x^{(1)})^T \theta \\
    \vdots \\
    (x^{(m)})^T \theta
\end{bmatrix} - \begin{bmatrix}
    y^{(1)} \\
    \vdots \\
    y^{(m)}
\end{bmatrix} \\[10pt]
&= \begin{bmatrix}
    h_\theta(x^{(1)}) - y^{(1)} \\
    \vdots \\
    h_\theta(x^{(m)}) - y^{(m)}
\end{bmatrix}
\end{aligned}
$$
</div>

这里 $ X \theta - \vec{y} $ 的大小是：$ m × 1 $。因为 $ \theta $ 是一个大小为 $ (n+1) × 1 $ 的列向量，所以 $ X \theta $ 的大小是：$ m × 1 $；进一步地，$ X \theta - \vec{y} $ 的大小也是：$ m × 1 $。

此时，我们的误差也就是 $ X \theta - \vec{y} $。

对于一个矩阵 $ z $，我们有性质：$ z^T z = \sum_i z_i^2 $，然后我们将误差转化为实数:
<div class="math">
$$
\begin{aligned}
\frac{1}{2} (X \theta - \tilde{y})^T (X \theta - \tilde{y}) &= \frac{1}{2} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 \ \color{red}{\rightarrow 代价函数 J(\theta) 的由来} \\[5pt] & = J(\theta)
\end{aligned}
$$
</div>

最后，为了最小化 $J$, 我们需要对 $ \theta $ 进行求导。由我们之前在有关 **[矩阵的迹](/note2/Part2?id=矩阵的迹)** 中提到的性质 $ (2)(3) $，有：

<div class="math">
$$
\nabla_{A^T} \text{tr} (ABA^T C) = B^T A^T C^T + BA^T C \tag{4}
$$
</div>

下面开始证明：

<div class="math">
$$
\begin{align*}
proof:\ & \nabla_\theta J(\theta) = X^TX\theta - X^T\vec{y} \\
& \begin{aligned}[t]
    \nabla_\theta J(\theta) &= \nabla_{\theta} \frac{1}{2} (X\theta - y)^T (X\theta - y) \\[5pt]
    &= \frac{1}{2} \nabla_{\theta} (\theta^T X^T X \theta - \theta^T X^T y - y^T X \theta + y^T y)\ \color{red}{\rightarrow矩阵转置的展开} \\[5pt]
    &= \frac{1}{2} \nabla_{\theta} \mathrm{tr} (\theta^T X^T X \theta {\color{blue}- \theta^T X^T y - y^T X \theta} + y^T y)\  \color{red}{\rightarrow J(\theta)是一个实数，有:\mathrm{tr}J = J} \\[5pt]
    &= \frac{1}{2} \nabla_{\theta} ({\color{green}\mathrm{tr} \theta^T X^T X \theta} {\color{green}- 2\vec{y}^TX\theta}) \ \color{red}{\rightarrow \mathrm{tr}A^T = \mathrm{tr}A} 与 迹的循环不变性 \\[5pt]
    With\ Equation (4):\\
& \nabla_\theta \mathrm{tr} \theta^T X^T X \theta = X^TX\theta + X^TX\theta \\[5pt]
     With\ Equation (1):\\
& \nabla_{A} \text{tr} AB = B^T \\[5pt]
& \nabla_\theta \mathrm{tr}\vec{y}^T X\theta = \vec{y}^T X\theta = (\vec{y}^T X)^T = X^T\vec{y}\\[5pt]
    Hence,\\
& \nabla_\theta J(\theta) = X^TX\theta - X^T\vec{y} := 0 \\[5pt]
& \Rightarrow \color{red}\theta = (X^TX)^{-1}X^T\vec{y}
\end{aligned}
\end{align*}
$$
</div>

由此，我们得到了 **正规方程 （normal equation）**，即

<div class="math">
$$
 \theta = (X^TX)^{-1}X^T\vec{y}
$$
</div>
