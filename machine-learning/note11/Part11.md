# 第十一章&ensp;PCA 主成分分析

在这一个章节里，我们会学习到如何去测试训练的模型，以及去修正模型的误差。

通过下面的学习，应该重点掌握：

* PCA 在降维的应用

- - -

在讨论因子分析时，我们提供了一种对数据 <span class="math">$ x \in \mathbb{R}^n $</span> 进行建模的方法，假设其"近似"位于某个 <span class="math">$ k $</span> 维子空间中，且 <span class="math">$ k \ll n $</span>。具体来说，我们假设每个点 <span class="math">$ x^{(i)} $</span> 是通过如下规则生成的：首先，在 <span class="math">$ k $</span> 维仿射空间 <span class="math">$ \\{ \Lambda z + \mu; z \in \mathbb{R}^k \\} $</span> 中生成某个 <span class="math">$ z^{(i)} $</span>，然后加上 <span class="math">$ \Psi- $</span> 协方差噪声。因子分析基于概率模型，其参数估计使用了迭代的 EM 算法。

接下来，我们将学习一种新的方法—— **主成分分析（Principal Components Analysis）**，该方法尝试识别出数据近似所在的子空间。然而，PCA 的学习算法将会更直接，并且只需要进行一个特征向量计算（可以轻松使用 Matlab 中的 eig 函数完成），无需借助 EM 算法。

为了举一个不太自然的例子，考虑一个从遥控直升机飞行员调查中得到的数据集，其中 <span class="math">$ x_1^{(i)} $</span> 是第 <span class="math">$ i $</span> 位飞行员的飞行技能的衡量指标，而 <span class="math">$ x_2^{(i)} $</span> 衡量的是该飞行员对飞行的喜爱程度。因为遥控直升机非常难以操作，只有最专注的学生，换句话说，那些真正喜欢飞行的人，才能成为优秀的飞行员。因此，这两个属性 <span class="math">$ x_1 $</span> 和 <span class="math">$ x_2 $</span> 是高度相关的。事实上，我们可以假设 <span class="math">$ x_1 $</span> 和 <span class="math">$ x_2 $</span> 是高度线性相关的。

数据实际上沿着某条对角轴（即 <span class="math">$ u_1 $</span> 方向）分布，这条轴捕捉了某个人固有的飞行操纵"能力"，只有少量的噪声偏离了这条轴。我们该如何自动计算出这个 <span class="math">$ u_1 $</span> 方向呢？

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/09/18/pAKuf3t.png" data-lightbox="image-11" data-title="PCA1">
  <img src="https://s21.ax1x.com/2024/09/18/pAKuf3t.png" alt="PCA1" style="width:100%;max-width:350px;cursor:pointer">
 </a>
</div>

在图中，横轴为 <span class="math">$ x_1 $</span> （飞行技能），纵轴为 <span class="math">$ x_2 $</span> （对飞行的喜好程度）。数据点沿着 <span class="math">$ u_1 $</span> 方向呈现线性分布， <span class="math">$ u_2 $</span> 则与 <span class="math">$ u_1 $</span> 垂直。

接下来我们将推导出 PCA 算法。但是，在运行 PCA 之前，我们通常会首先对数据进行预处理，以对其均值和方差进行归一化处理，步骤如下：

> [!NOTE]
> **PCA 数据预处理步骤：**
>
> <span class="math">$ 1.\ 设\ \mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)} $</span>
>
> <span class="math">$ 2.\ 用\ x^{(i)} - \mu\ 替换每个\ x^{(i)} $</span>
>
> <span class="math">$ 3.\ 设\ \sigma_j^2 = \frac{1}{m} \sum_{i} (x_j^{(i)})^2 $</span>
>
> <span class="math">$ 4.\ 用\ x_j^{(i)} / \sigma_j\ 替换每个\ x_j^{(i)} $</span>

步骤<span class="math">$(1-2)$</span>将数据的均值归零，这对那些已知具有零均值的数据（例如，一些与语音或其他声学信号相关的时间序列数据）可以省略。步骤<span class="math">$(3-4)$</span>对每个坐标进行重新缩放，使其具有单位方差，这确保了不同的属性在相同的“尺度”上进行处理。例如，如果 <span class="math">$ x_1 $</span> 是汽车的最高速度（以 mph 为单位，数值在十几到一百多之间），而 <span class="math">$ x_2 $</span> 是座位数（数值大约为 2-4），那么这种重新归一化会将不同属性进行调整，使它们更具有可比性。如果我们事先知道不同属性在相同的尺度上，也可以省略步骤<span class="math">$(3-4)$</span>。例如，如果每个数据点代表一幅灰度图像，并且每个 <span class="math">$ x_j^{(i)} $</span> 的取值范围是 <span class="math">$ \\{0, 1, \ldots, 255\\} $</span>，对应于图像 <span class="math">$ i $</span> 中像素 <span class="math">$ j $</span> 的强度值。

现在，在完成了归一化之后，我们如何计算"主要变化轴" <span class="math">$ u $</span> ——也就是数据大致所在的方向？一种解决这个问题的方法是找到单位向量 <span class="math">$ u $</span>，使得当数据投影到与 <span class="math">$ u $</span> 对应的方向时，投影数据的方差达到最大。直观上，数据开始时包含一定的方差/信息量。我们希望选择一个方向 <span class="math">$ u $</span>，这样当我们近似数据时，使其位于与 <span class="math">$ u $</span> 对应的方向/子空间中，能够保留尽可能多的方差。

考虑下列数据集，我们已经对其进行了归一化步骤：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/09/18/pAKu4jf.png" data-lightbox="image-11" data-title="PCA2">
  <img src="https://s21.ax1x.com/2024/09/18/pAKu4jf.png" alt="PCA2" style="width:100%;max-width:350px;cursor:pointer">
 </a>
</div>

现在，假设我们选择了 <span class="math">$ u $</span> 对应于图中所示的方向。圆圈表示原始数据投影到该条线上的点。

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/09/18/pAKuIu8.png" data-lightbox="image-11" data-title="PCA3">
  <img src="https://s21.ax1x.com/2024/09/18/pAKuIu8.png" alt="PCA3" style="width:100%;max-width:350px;cursor:pointer">
 </a>
</div>

我们看到，投影后的数据仍然具有相当大的方差，且数据点往往离零点较远。相比之下，假设我们选择了如下图所示的方向：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/09/18/pAKuoDS.png" data-lightbox="image-11" data-title="PCA4">
  <img src="https://s21.ax1x.com/2024/09/18/pAKuoDS.png" alt="PCA4" style="width:100%;max-width:350px;cursor:pointer">
 </a>
</div>

在这里，投影后的数据明显具有更小的方差，并且距离原点更近。

我们希望能够自动选择与上面第一个图对应的方向 <span class="math">$ u $</span>。为了形式化这一点，给定一个单位向量 <span class="math">$ u $</span> 和点 <span class="math">$ x $</span>，则 <span class="math">$ x $</span> 在 <span class="math">$ u $</span> 上的投影长度为 <span class="math">$ x^T u $</span>。即，如果 <span class="math">$ x^{(i)} $</span> 是我们数据集中的一个点（图中的一个画叉的点），那么它在 <span class="math">$ u $</span> 上的投影（对应图中的圆点）到原点的距离就是 <span class="math">$ x^T u $</span>。因此，为了最大化投影的方差，我们需要选择一个单位长度向量 <span class="math">$ u $</span>，使得：

<div class="math">
$$
\begin{aligned}
\frac{1}{m} \sum_{i=1}^{m} \left( (x^{(i)})^T u \right)^2 &= \frac{1}{m} \sum_{i=1}^{m} u^T x^{(i)} (x^{(i)})^T u \\[5pt]
&= u^T \left( \frac{1}{m} \sum_{i=1}^{m} x^{(i)} (x^{(i)})^T \right)
\end{aligned}
$$
</div>

我们很容易意识到，最大化此值的前提是 <span class="math">$ \lVert u \rVert_2 = 1 $</span>，它给出了矩阵 <span class="math">$ \Sigma = \frac{1}{m} \sum_{i=1}^{m} x^{(i)} (x^{(i)})^T $</span> 的主特征向量，其中 <span class="math">$ \Sigma $</span> 是数据的经验协方差矩阵（假设它具有零均值）。

> [!NOTE]
> 如果以前没见过这种形式，可以用拉格朗日数法将 <span class="math">$ u^T \Sigma u $</span> 最大化，使得 <span class="math">$ u^T u = 1 $</span>。你应该能发现对于某些 <span class="math">$ \lambda $</span>，<span class="math">$ \Sigma u = \lambda u $</span>，这就意味着向量 <span class="math">$ u $</span> 是 <span class="math">$ \Sigma $</span> 的特征向量，特征值为 <span class="math">$ \lambda $</span>。

总的来说，我们发现，如果我们希望找到 1 维子空间来近似数据，我们应该选择 <span class="math">$ u $</span> 为 <span class="math">$ \Sigma $</span> 的主特征向量。更一般地说，如果我们希望将数据投影到一个 <span class="math">$ k $</span> 维子空间（其中 <span class="math">$ k < n $</span>），我们应该选择 <span class="math">$ u_1, u_2, \ldots, u_k $</span> 为 <span class="math">$ \Sigma $</span> 的前 <span class="math">$ k $</span> 个特征向量。这些 <span class="math">$ u_i $</span> 现在构成了数据的新正交基。

> [!NOTE]
> 由于 <span class="math">$ \Sigma $</span> 是对称的，所以向量 <span class="math">$ u_i $</span> 总是（或是总是能选出来）彼此正交的。

然后，要使用这组正交基来表示 <span class="math">$ x^{(i)} $</span>，只需计算相应的向量：

<div class="math">
$$
y^{(i)} = \begin{bmatrix}
u_1^T x^{(i)} \\
u_2^T x^{(i)} \\
\vdots \\
u_k^T x^{(i)}
\end{bmatrix} \in \mathbb{R}^k
$$
</div>

因此，虽然 <span class="math">$ x^{(i)} \in \mathbb{R}^n $</span>，但向量 <span class="math">$ y^{(i)} $</span> 现在得到了由 <span class="math">$ x^{(i)} $</span> 表示的一个更低维的 <span class="math">$ k $</span> 维近似/表示。因此，PCA 也被称为一种 **降维（dimensionality reduction）** 算法。向量 <span class="math">$ u_1, \ldots, u_k $</span> 称为数据的前 <span class="math">$ k $</span> 个 **主成分（principal components）**。

> [!NOTE]
> 尽管我们形式上只展示了 <span class="math">$ k = 1 $</span> 的情况，但利用特征向量的已知性质，可以很容易地推广到一般情况。在所有可能的正交基 <span class="math">$ u_1, \ldots, u_k $</span> 中，我们选择的基最大化了 <span class="math">$ \sum_i \|\|y^{(i)}\|\|_2^2 $</span>。因此，我们选择的基需要尽可能多地保留了原始数据中的方差信息。

此外，PCA 还可以通过选择最小化近似误差的基得出，该误差源自将数据投影到它们所张成的 <span class="math">$ k $</span> 维子空间上。

PCA有许多应用；我们将通过几个例子来结束讨论。首先，压缩——用较低维度的 <span class="math">$ y^{(i)} $</span> 表示 <span class="math">$ x^{(i)} $</span>。如果我们将高维数据降到 <span class="math">$ k = 2 $</span> 或 <span class="math">$ 3 $</span> 维，然后我们还可以绘制 <span class="math">$ y^{(i)} $</span> 来可视化数据。例如，如果我们将汽车数据降到二维，那么我们可以绘制出每个点（在我们的图中，每个点可能代表一种车型），从而查看哪些汽车彼此相似，哪些汽车可能会聚集在一起。

另一个标准应用是预处理数据集，降低其维度，以便在进行监督学习算法时将 <span class="math">$ x^{(i)} $</span> 用作输入。除了计算效益外，减少数据的维度还可以降低假设类的复杂性，并帮助避免过拟合（例如，低维输入空间上的线性分类器将具有较小的 VC 维度）。
