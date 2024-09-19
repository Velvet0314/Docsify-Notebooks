# 第十二章&ensp;ICA 独立成分分析

在这一章节中，我们将学习 **独立成分分析（Independent Components Analysis）**。与主成分分析（PCA）相似，这将为我们提供一种新的数据表示方式。然而，ICA 的目标却与 PCA 有很大不同。

下面，我们介绍一个称为"鸡尾酒会问题"的例子。在这个例子中，有多个说话者同时在派对上讲话，并且一个麦克风记录下来的一定只是这些说话者声音的叠加混合。但假设我们有 <span class="math">$n$</span> 个不同位置的麦克风，并且由于每个麦克风到各个说话者的距离不同，每个麦克风记录下的声音都是这些说话者声音的不同组合。使用这些麦克风录音，我们能否从中分离出每个说话者的语音信号呢？

为了形式化这一问题，我们设想存在某些数据 <span class="math">$s \in \mathbb{R}^n$</span>，它是由 <span class="math">$n$</span> 个独立源生成的。我们观察到的数据是：

<div class="math">
$$
x = As,
$$
</div>

其中 <span class="math">$A$</span> 是一个未知的方阵，称为**混合矩阵（mixing matrix）**。通过反复地观测，我们得到了一个数据集 <span class="math">$\\{x^{(i)}: i = 1, \dots, m\\}$</span>，我们的目标是恢复这些源信号 <span class="math">$s^{(i)}$</span>，即生成我们数据的源信号 (<span class="math">$x^{(i)} = As^{(i)}$</span>)。

在我们的鸡尾酒会问题中，<span class="math">$s^{(i)}$</span> 是一个 <span class="math">$n$</span> 维向量，<span class="math">$s_j^{(i)}$</span> 是在时间 <span class="math">$t$</span> 时刻说话者 <span class="math">$j$</span> 的发声。同样，<span class="math">$x^{(i)}$</span> 是一个 <span class="math">$n$</span> 维向量，<span class="math">$x_j^{(i)}$</span> 是在时间 <span class="math">$t$</span> 时刻由麦克风 <span class="math">$j$</span> 记录的声学信号。

令 <span class="math">$W = A^{-1}$</span> 是 **解混矩阵（unmixing matrix）**。我们的目标是找到 <span class="math">$W$</span>，从而利用麦克风录音 <span class="math">$x^{(i)}$</span>，通过以下公式恢复源信号：

<div class="math">
$$
s^{(i)} = Wx^{(i)}
$$
</div>

为了方便表示，我们还让 <span class="math">$w_i^T$</span> 表示矩阵 <span class="math">$W$</span> 的第 <span class="math">$i$</span> 行，因此有：

<div class="math">
$$
W = \begin{bmatrix}
    - w_1^T - \\
    \vdots \\
    - w_n^T -
\end{bmatrix}
$$
</div>

于是我们有 <span class="math">$w_i \in \mathbb{R}^n$</span>，进而第 <span class="math">$j$</span> 个源信号可以通过计算 <span class="math">$s_j^{(i)} = w_j^T x^{(i)}$</span> 来恢复了。

通过下面的学习，应该重点掌握：

* ICA 算法

- - -

### ICA 的模糊性

<span class="math">$ W = A^{-1} $</span> 能够恢复到什么程度呢？如果我们对源信号和混合矩阵没有任何先验知识（预先的了解），那么很容易看出，如果只是通过 <span class="math">$x^{(i)}$</span>，<span class="math">$A$</span> 中的一些固有的模糊是无法解决的。

具体来说，设 <span class="math">$P$</span> 为任意 <span class="math">$n \times n$</span> 的置换矩阵。这意味着 <span class="math">$P$</span> 的每一行和每一列都有且仅有一个"1"。下面是一些置换矩阵的例子：

<div class="math">
$$
P = \begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}\text{；}
P = \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}\text{；}
P = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$
</div>

如果 <span class="math">$z$</span> 是一个向量，那么 <span class="math">$Pz$</span> 就是包含 <span class="math">$z$</span> 的坐标置换后的另一个向量。只给定 <span class="math">$x^{(i)}$</span>，将无法区分 <span class="math">$W$</span> 和 <span class="math">$PW$</span>。具体来说，源信号的置换是不确定的，这并不奇怪。幸运的是，这对于大多数应用来说并不重要。

也就是说，没有办法恢复 <span class="math">$w_i$</span> 的正确缩放。举例来说，如果 <span class="math">$A$</span> 被替换为 <span class="math">$2A$</span>，并且 <span class="math">$s^{(i)}$</span> 被替换为 <span class="math">$0.5s^{(i)}$</span>，那么我们观察到的 <span class="math">$x^{(i)} = 2A \cdot (0.5)s^{(i)}$</span> 将保持不变。广义上讲，如果 <span class="math">$A$</span> 的某一列被按因子 <span class="math">$\alpha$</span> 缩放，而相应的源信号则按 <span class="math">$1/\alpha$</span> 缩放，那么单从 <span class="math">$x^{(i)}$</span> 来看，仍然无法判断出这种变化。因此，我们无法恢复出源信号的"正确"缩放。然而，对于我们关心的应用——包括鸡尾酒会问题——这种不确定性同样也不重要。具体来说，按某个正的因子缩放说话者的语音信号 <span class="math">$s_j^{(i)}$</span> 只会影响该说话者语音的音量。同样，符号的变化也不重要，因为 <span class="math">$s_j^{(i)}$</span> 和 <span class="math">$-s_j^{(i)}$</span> 在播放时听起来是一样的，都在扬声器中是同样的音量大小。因此，如果通过算法找到的 <span class="math">$w_i$</span> 被任意非零实数缩放，相应恢复出来的源信号 <span class="math">$s_i = w_i^T x$</span> 也会按同样的因子缩放；但这通常并不重要。

上面这些是 ICA 唯一模糊性来源吗？事实证明，只要源信号 <span class="math">$s_i$</span> 是非高斯分布的，就没有其他的不确定性因素。为了说明高斯分布下数据中的困难情况，考虑一个例子，其中 <span class="math">$n = 2$</span>，并且 <span class="math">$s \sim \mathcal{N}(0, I)$</span>，其中 <span class="math">$I$</span> 是 <span class="math">$2 \times 2$</span> 单位矩阵。注意标准正态分布 <span class="math">$\mathcal{N}(0, I)$</span> 的密度轮廓是以原点为中心的圆，并且密度是旋转对称的。

现在，假设我们观察到 <span class="math">$x = As$</span>，其中 <span class="math">$A$</span> 是我们的混合矩阵。<span class="math">$x$</span> 的分布也将是高斯分布，均值为零且协方差 <span class="math">$E[xx^T] = E[Ass^T A^T] = AA^T$</span>。现在，设 <span class="math">$R$</span> 为任意正交矩阵（形式上是旋转/反射矩阵），使得 <span class="math">$RR^T = R^T R = I$</span>，并设 <span class="math">$A' = AR$</span>。如果数据是通过 <span class="math">$A$</span> 混合的，而不是通过 <span class="math">$A'$</span> 混合的，我们将观察到 <span class="math">$x' = A's$</span>。<span class="math">$x'$</span> 的分布仍然是高斯分布，均值为零且协方差 <span class="math">$E[x'(x')^T] = E[Ass^T(A')^T] = E[ARss^T(AR)^T] = ARR^TA^T= AA^T$</span>。因此，我们将从 <span class="math">$\mathcal{N}(0, AA^T)$</span> 分布中观察数据。因此，无法从中分辨出数据是使用 <span class="math">$A$</span> 还是 <span class="math">$A'$</span> 混合的。换句话说，只要混合矩阵中存在一个任意的旋转分量，无法通过数据推导出来，我们就无法恢复原始源信号。

我们的论点是基于多元标准正态分布是旋转对称的这个定理。这些情况使得 ICA 面对高斯分布的数据的时候很无力，但是只要数据不是高斯分布的，然后再有充足的数据，那就还是能恢复出 n 个独立的声源的。

### 密度和线性变换

在正式推导 ICA 算法之前，我们先简要讨论一下线性变换对密度函数的影响。

假设我们有一个随机变量 <span class="math">$s$</span>，其密度分布由 <span class="math">$p_s(s)$</span> 绘制。为了简化讨论，暂时假设 <span class="math">$s \in \mathbb{R}$</span> 是一个实数。现在，定义随机变量 <span class="math">$x = As$</span>（这里 <span class="math">$x \in \mathbb{R}$</span>，<span class="math">$A \in \mathbb{R}$</span>）。设 <span class="math">$p_x$</span> 是 <span class="math">$x$</span> 的密度。那么，现在 <span class="math">$p_x$</span> 是多少呢？

令 <span class="math">$W = A^{-1}$</span>。为了计算 <span class="math">$x$</span> 取某一特定值的"概率"，我们可能会尝试先计算 <span class="math">$s = Wx$</span>，然后在该点计算 <span class="math">$p_s$</span>，并得出 "<span class="math">$p_x(x) = p_s(Wx)$</span>"。然而，这是不正确的。例如，设 <span class="math">$s \sim \text{Uniform}[0, 1]$</span>，因此 <span class="math">$s$</span> 的密度为 <span class="math">$p_s(s) = 1\\{0 \leq s \leq 1\\}$</span>。现在，令 <span class="math">$A = 2$</span>，因此 <span class="math">$x = 2s$</span>。显然，<span class="math">$x$</span> 在区间 <span class="math">$[0, 2]$</span> 上均匀分布。因此，<span class="math">$x$</span> 的密度为 <span class="math">$p_x(x) = (0.5)1\\{0 \leq x \leq 2\\}$</span>。这与 <span class="math">$p_s(Wx)$</span>，其中 <span class="math">$W = 0.5 = A^{-1}$</span>，不相等。正确的公式应该是：

<div class="math">
$$
p_x(x) = p_s(Wx) \cdot |W|
$$
</div>

更一般地，如果 <span class="math">$s$</span> 是一个具有密度 <span class="math">$p_s$</span> 的向量值分布，而 <span class="math">$x = As$</span>，其中 <span class="math">$A$</span> 是一个可逆方阵，那么 <span class="math">$x$</span> 的密度函数为：

<div class="math">
$$
p_x(x) = p_s(Wx) \cdot |W|
$$
</div>

其中 <span class="math">$W = A^{-1}$</span>。

> [!NOTE]
> 可能你已经看到 <span class="math">$A$</span> 将 <span class="math">$[0,1]^n$</span> 映射到体积 <span class="math">$|A|$</span> 的集合，那么还有另一种记住上述 <span class="math">$p_x$</span> 公式的方法，这种方法可以推广到我们的前述一维示例。具体来说，令 <span class="math">$A \in \mathbb{R}^{n \times n}$</span> 已知，且 <span class="math">$W = A^{-1}$</span> 也已知。令 <span class="math">$C_1 = [0,1]^n$</span> 为 <span class="math">$n$</span> 维超立方体，定义 <span class="math">$C_2 = \{As : s \in C_1\} \subseteq \mathbb{R}^n$</span> 为通过映射 <span class="math">$A$</span> 得到的 <span class="math">$C_1$</span> 的像。那么，在线性代数中这是一个标准结果（实际上，这也是定义行列式的多种方式之一），即 <span class="math">$C_2$</span> 的体积由 <span class="math">$|A|$</span> 给出。现在，假设 <span class="math">$s$</span> 均匀分布在 <span class="math">$[0, 1]^n$</span>，则其密度为 <span class="math">$p_s(s) = 1\{s \in C_1\}$</span>。那么 <span class="math">$x$</span> 显然也均匀分布在 <span class="math">$C_2$</span> 中。因此，<span class="math">$x$</span> 的密度应为：
>
> <div class="math">
> $$
> p_x(x) = \frac{1\{x \in C_2\}}{\text{vol}(C_2)} = \frac{1\{x \in C_2\}}{|A|}
> $$
> </div>
>
> 这是因为体积的确定性是由矩阵 <span class="math">$A$</span> 的行列式给出的，而 <span class="math">$1/\text{vol}(C_2) = 1/|A| = |W|$</span>。因此：
>
> <div class="math">
> $$
> p_x(x) = 1\{x \in C_2\} |W| = 1\{Wx \in C_1\} |W| = p_s(Wx) |W|
> $$
> </div>

### ICA 算法

我们现在准备推导 ICA 算法。我们描述的算法源自于 Bell 和 Sejnowski，他们对该算法的解释是将其视为一种最大似然估计。（这与他们最初的解释不同，最初的解释涉及到一个叫做信息最大化原理的复杂概念，这在现代 ICA 理解中已经不再需要。）

我们假设每个源信号 <span class="math">$s_i$</span> 的分布由密度 <span class="math">$p_{s_i}$</span> 给出，并且源信号 <span class="math">$s$</span> 的联合分布为：

<div class="math">
$$
p(s) = \prod_{i=1}^{n} p_{s_i}(s_i)
$$
</div>

注意，通过将联合分布建模为边缘分布的乘积，我们能得到源信号是独立的。利用我们在前一节中的公式，这意味着对于 <span class="math">$x = As = W^{-1} s$</span>，其密度函数为：

<div class="math">
$$
p(x) = \prod_{i=1}^{n} p_{s_i}(w_i^T x) \cdot |W|
$$
</div>

接下来就是为每个独立的源信号 <span class="math">$p_{s_i}$</span> 指定一个对应的密度函数。

回想一下，给定一个实值随机变量 <span class="math">$z$</span>，它的累积分布函数 <span class="math">$(cdf)$</span> <span class="math">$F$</span> 定义为：

<div class="math">
$$
F(z_0) = P(z \leq z_0) = \int_{-\infty}^{z_0} p_z(z) dz
$$
</div>

同时，<span class="math">$z$</span> 的密度函数可以通过取其 <span class="math">$cdf$</span> 的导数来求出：<span class="math">$p_z(z) = F'(z)$</span>。

因此，为了为 <span class="math">$s_i$</span> 指定一个密度，我们只需要指定其 <span class="math">$cdf$</span>。<span class="math">$cdf$</span> 必须是从 0 增加到 1 的单调函数。根据我们之前的讨论，我们不能选择高斯分布的 <span class="math">$cdf$</span>，因为 ICA 不能处理高斯数据。相反，作为一个合理的"默认"函数，我们选择随着 0 到 1 缓慢增长的 <span class="math">$Sigmoid$</span> 函数 <span class="math">$\displaystyle{g(s) = \frac{1}{1 + e^{-s}}}$</span>。因此，<span class="math">$p_{s_i}(s) = g'(s)$</span>。

> [!NOTE]
> 如果你对源信号的密度形式有过了解，那么在这里可以对概念进行等价替换。但如果事先没有了解过，那么 <span class="math">$Sigmoid$</span> 函数可以被认为是一个合理的默认选择，并且在许多问题中表现良好。此外，这里的推导假设数据 <span class="math">$x^{(i)}$</span> 要么已经经过预处理为零均值，要么可以自然地预期为零均值（例如声学信号）。这是必要的，因为我们假设 <span class="math">$p_s(s) = g'(s)$</span> 也就蕴含了 <span class="math">$E[s] = 0$</span>（逻辑函数的导数是对称函数，因此对应于零均值随机变量的密度），这进一步意味着 <span class="math">$E[x] = E[As] = 0$</span>。

方阵 <span class="math">$W$</span> 是我们模型中的参数。给定一个训练集 <span class="math">$\\{x^{(i)} : i = 1, \dots, m\\}$</span>，其对数似然函数为：

<div class="math">
$$
\ell(W) = \sum_{i=1}^{m} \left( \sum_{j=1}^{n} \log g'(w_j^T x^{(i)}) + \log |W| \right)
$$
</div>

我们希望通过 <span class="math">$W$</span> 的梯度最大化该函数。通过取导数并利用 <span class="math">$\nabla_W |W| = |W| (W^{-1})^T$</span>（该公式在前面的笔记中给出），我们很容易导出随机梯度下降学习规则。对于一个训练样本 <span class="math">$x^{(i)}$</span>，更新规则为：

<div class="math">
$$
W = W + \alpha \left( \begin{bmatrix}
1 - 2g(w_1^T x^{(i)}) \\
1 - 2g(w_2^T x^{(i)}) \\
\vdots \\
1 - 2g(w_n^T x^{(i)})
\end{bmatrix}
x^{(i)T} + (W^T)^{-1} \right)
$$
</div>

其中 <span class="math">$\alpha$</span> 是学习率。

在算法收敛后，我们可以计算 <span class="math">$s^{(i)} = W x^{(i)}$</span> 来恢复原始的源信号。

> [!NOTE]
> 当我们写下数据的似然估计时，我们隐含假设了不同的 <span class="math">$x^{(i)}$</span> 彼此独立（对于不同的 <span class="math">$i$</span> 值；注意这个问题不同于 <span class="math">$x^{(i)}$</span> 的不同坐标是否独立），因此训练集的似然由 <span class="math">$\prod_i p(x^{(i)}; W)$</span> 给出。这个假设显然对语音数据和其他时间序列数据不正确，因为 <span class="math">$x^{(i)}$</span> 是相关的，但可以证明，即使训练样本之间存在相关性，算法的性能也不会受到影响，只要我们有足够的数据。然而，对于那些连续的训练样本彼此相关的问题，在执行随机梯度上升时，随机访问训练样本也有助于加速收敛。也就是说，执行随机梯度上升时，我们可以在随机打乱的训练集副本上运行该算法。
