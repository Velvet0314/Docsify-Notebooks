# 第二章&ensp;线性回归

在本章中，我们将学习线性回归。线性回归是绝大多数机器学习中的第一课，同时也是与我们过去累积到的经验最相似的一部分。

通过下面的学习，应该对以下知识有着基本的了解：

* BSD 与 SGD 的区别
* SDG 中存在收敛振荡
* 局部加权线性回归的原理

通过下面的学习，应该重点掌握：

* 线性回归的原理
* 假设函数与代价函数
* 梯度下降
* 正规方程及其推导

为了易于后续公式表达，引入以下符号：

* $x\_j^{(i)}$：第 $i$ 个训练样本的特征 $j$ 的值
* $x^{(i)}$：第 $i$ 个训练样本的所有特征
* $m$：训练样本的数量
* $n$：特征的数量

- - -

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
h_{\theta}(x) = \big[\theta_0,\ \theta_1,\ ... ,\ \theta_n\big] \left[\begin{matrix}x_0\\\ x_1\\\ ...\\\ x_n\end{matrix}\right]= \theta^Tx
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

### ⭐LMS Algorithm

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

#### ⭐小结：LMS

##### BGD 与 SGD 的区别

批量梯度下降法（BGD）和随机梯度下降法（SGD）是都采用了LMS的核心思想，但其区别在于：

* 批量梯度下降法每次更新参数时，使用全部的训练样本。
* 随机梯度下降法每次更新参数时，使用一个随机的训练样本。

如果训练集过大，一般采用 SGD 以加速收敛。

##### SGD 与 学习率衰减

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

将每一个参数 $ \theta $ 对应的 $ x $ 值看做是一个列向量，那么特征矩阵 $ X $ （design matrix）可以表示为：

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
    &= \frac{1}{2} \nabla_{\theta} \mathrm{tr} (\theta^T X^T X \theta {\color{orange}- \theta^T X^T y - y^T X \theta} + y^T y)\  \color{red}{\rightarrow J(\theta)是一个实数，有:\mathrm{tr}J = J} \\[5pt]
    &= \frac{1}{2} \nabla_{\theta} ({\color{lightgreen}\mathrm{tr} \theta^T X^T X \theta} {\color{lightgreen}- 2\vec{y}^TX\theta}) \ \color{red}{\rightarrow \mathrm{tr}A^T = \mathrm{tr}A} 与 迹的循环不变性 \\[5pt]
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

### 概率解释

除开数学上的推导，我们对于回归问题，还可以从概率的角度进行解释。

首先，我们假设目标值与输入通过以下的等式相关联：

<div class="math">
$$
y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}
$$
</div>

其中，$ \epsilon $ 是一个误差项，一般指未被学习到的要素或随机噪声。

进一步地，我们假设 $ \epsilon^{(i)} $ 是 $ distributed\ i.i.d. $ (离散式独立同分布变量) 并服从均值为 $ 0 $，方差为 $ \sigma^2 $ 的正态分布（高斯分布），即 $ \epsilon^{(i)} \sim N(0, \sigma^2) $ 。那么其概率密度函数为：

<div class="math">
$$
p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)
$$
</div>

如此，带入一开始的等式，我们有：
<div class="math">
$$
p(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(- \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)
$$
</div>

也就是说，在给定的参数 $\theta $ 下，确定了一个输入值 $ x^{(i)} $，得到对应的 $ y^{(i)} $ 的概率。

> [!TIP]
> **这里解释一下为什么公式没有写为 $ p(y^{(i)} | x^{(i)}, \theta) $ ：因为 $ \theta $ 并非一个随机变量**

在矩阵形式下，我们可以将概率写为：

<div class="math">
$$
p(\vec{y} \mid X ; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(- \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)
$$
</div>

在这样的一个形式下，我们将其视为一个 $ \theta $ 的函数：

<div class="math">
$$
L(\theta) = L(\theta; X, \vec{y}) = p(\vec{y} \mid X; \theta)
$$
</div>

这个函数被称为 **似然函数（likelihood function）**。

> [!NOTE]
> **⭐极大似然估计（maximum likelihood estimation）：** <br>
> **若总体 $ X $ 为离散型，$ \theta \in \Theta $，其中 $ \theta $ 为为待估参数，$ \Theta $ 是 $ \theta $ 可能取值范围。设 $ X_1,X_2,...,X_n $ 是来自 $ X $ 的样本，则其联合分布律为 $$ \prod_{i=1}^n p(x_{i}; \theta) $$** <br>
> **又设 $ x_1,x_2,...,x_n $ 是相应于样本 $ X_1,X_2,...,X_n $ 的一个样本值，则样本 $ X_1,X_2,...,X_n $ 取到观察值 $ x_1,x_2,...,x_n $ 的概率，即事件 $ \\{ X_1=x_1,X_2=x_2,...,X_n =x_n\\} $ 发生的概率为**
> **$$ L(\theta) = L(x_1,x_2,...,x_n;\theta) = \prod_{i=1}^n p(x_{i}; \theta),\ \theta \in \Theta $$**<br>
> **固定样本观察值 $ x_1,x_2,...,x_n $，在 $ \theta $ 取值的可能范围 $ \Theta $ 内挑选使似然函数达到最大的参数值 $ \hat{\theta} $，即：**<br>
> **$$ L(x_1,x_2,...,x_n;\hat{\theta}) = \max_{\theta \in \Theta} L(x_1,x_2,...,x_n;\theta)$$**

基于 $ \epsilon^{(i)} $ 的独立同分布，我们可以得到：

<div class="math">
$$
\begin{aligned}
L(\theta) &= \prod_{i=1}^m p(y^{(i)} \mid x^{(i)}; \theta) \\
&= \prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{\left(y^{(i)} - \theta^T x^{(i)}\right)^2}{2\sigma^2} \right)
\end{aligned}
$$
</div>

为了选择合适的 $ \theta $ 以最大化 $ L(\theta) $，采用 **极大似然估计** 的思想，并且将其转化为 **对数似然函数** 易于后续计算：

<div class="math">
$$
\begin{aligned}
\ell(\theta) &= \log L(\theta) \\
&= \log \prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{\left(y^{(i)} - \theta^T x^{(i)}\right)^2}{2\sigma^2} \right)
\end{aligned}
$$
</div>

如此，最大化 $ \ell(\theta) $ 就是去最小化

<div class="math">
$$ 
 \frac{1}{2} \sum_{i=1}^m \left(y^{(i)} - \theta^T x^{(i)}\right)^2
$$
</div>

同样的，这也就是我们的 **[代价函数](note2/Part2?id=代价函数)** 。

### 局部加权线性回归

#### 过拟合与欠拟合

下面是一个关于 **欠拟合（underfitting）** 和 **过拟合（overfitting）** 的很形象的例子：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/05/22/pkMWmz8.png" data-lightbox="image-2" data-title="Example of underfitting and overfitting">
  <img src="https://s21.ax1x.com/2024/05/22/pkMWmz8.png" alt="Example of underfitting and overfitting" style="width:100%;max-width:500px;cursor:pointer">
 </a>
</div>

下面是一个具体训练模型可视化的例子：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/05/22/pkMg0eA.jpg" data-lightbox="image-2" data-title="fitting in Machine Learning">
  <img src="https://s21.ax1x.com/2024/05/22/pkMg0eA.jpg" alt="fitting in Machine Learning" style="width:100%;max-width:2000px;cursor:pointer">
 </a>
</div>

在这个例子中我们可以看到，在模型拟合的多项式次数较低时（图中展示的为1次），函数呈线性，拟合效果不好；而当模型拟合的多项式次数较高时（图中展示的为20次），函数学习到了过多的细节与噪声，完全适合与当前的训练集，但是这样其泛用性会降低，也并非一个好的模型；在一个适当的多项式次数下（图中展示的为4次），拟合效果不错。

#### LWLR 的定义

经典的线性回归是一种无偏差估计，即优化策略过于整体，易发生欠拟合，故我们采用 **局部加权线性回归（locally weighted linear regression）**。以下是两者学习过程上的区别：

<div class="math">
$$
\begin{aligned}
&\text{对于一般的线性回归：}\\[5pt]
&\ \ \ \ \ \ \ \ \ \ 1.\ Fit \ \theta \ to \ minimize\ \sum\nolimits_{i} (y^{(i)} - \theta^T x^{(i)})^2 \\
&\ \ \ \ \ \ \ \ \ \ 2.\ Output \ \theta^T x \\[5pt]
&\text{对于局部加权线性回归：}\\[5pt]
&\ \ \ \ \ \ \ \ \ \ 1.\ Fit \ \theta \ to \ minimize\ \sum\nolimits_{i} w^{(i)} (y^{(i)} - \theta^T x^{(i)})^2 \\
&\ \ \ \ \ \ \ \ \ \ 2.\ Output \ \theta^T x \\
\end{aligned}
$$
</div>

其中，$ w $ 一般通过以下公式得到：

<div class="math">
$$
w^{(i)} = \exp \left( -\frac{(x^{(i)}-x)^2}{2\tau^2} \right)
$$
</div>

其中，$ x $ 是我们希望的预测点，$ \tau $ 是 **带宽参数（bandwidth）**，控制了权重随距离衰减的速度。

公式的形式类似于等质量万有引力定律。通过分析公式，我们发现：$ \left| x^{(i)}-x \right| $ 越大，$ w^{(i)} $ 越小；$ \left| x^{(i)}-x \right| $ 越小。
这表明：**离预测点越远的数据带来的影响会更小**。在 LWLR 中我们更关注预测点附近值带来的影响。

当 $ \tau \rightarrow \infty,w^{(i)} \rightarrow 1 $ 时，整个模型趋于标准的线性回归。