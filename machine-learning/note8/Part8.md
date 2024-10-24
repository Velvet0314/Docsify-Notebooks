# 第八章&ensp;正则化与模型选择

在这一个章节里，我们会学习到如何去测试训练的模型，以及如何去修正模型的误差。

通过下面的学习，应该重点掌握：

* 交叉验证
* 前向搜索

- - -

现在，假设我们正在尝试从不同种类的学习模型中选择一个合适的模型来解决当前的问题。例如，我们可能使用一个多项式回归模型 <span class="math">$ h_\theta(x) = g(\theta_0 + \theta_1 x + \theta_2 x^2 + ... + \theta_k x^k) $</span>，并且想要得到 <span class="math">$ k $</span> 的值应该是 <span class="math">$ 0,1,...\ \text{or}\ 10 $</span>。我们如何自动选择一个能在偏差和方差之间做出良好权衡的模型？或者说，假设我们想自动选择局部加权回归的带宽参数 <span class="math">$ \tau $</span>、<span class="math">$ \mathcal{l}_1 $</span>-正则化下的 SVM 中的参数 <span class="math">$ C $</span>，我们该如何操作来实现呢？

为了简单起见，在下面的讲解中，我们假设我们有一些模型集合 <span class="math">$ \mathcal{M} = \\{M_1, ..., M_d\\} $</span>，我们正尝试选择其中的一个。例如，在我们第一个例子中，模型 <span class="math">$ M_i $</span> 将是一个 <span class="math">$ i $</span>-阶多项式回归模型（将 <span class="math">$ M $</span> 扩展到无穷集合并不难）。另外，如果我们正在考虑使用支持向量机、神经网络或逻辑回归，那么 <span class="math">$ M $</span> 应该是包含了这些模型的。

> [!NOTE]
> 如果我们要从一个无穷的模型集合中进行选取一个，假如说要选取一个带宽参数 <span class="math">$ \tau \in \mathbb{R}^+ $</span>（正实数）的某个可能的值，可以将 <span class="math">$ \tau $</span> 离散化，而只考虑有限的一系列值。更广泛来说，我们要讲到的大部分算法都可以看做在 **模型空间（space of models）** 中进行 **优化搜索（performing optimization search）** 的问题，这种搜索也可以在 **无穷模型类（infinite model classes）** 上进行。

### 交叉验证

假设我们有一个训练集 <span class="math">$ S $</span>。基于经验风险最小化（ERM），我们将得到下面一个看似合理但实际不可行的算法，这是基于 ERM 进行模型选择所得到的结果：

> [!TIP]
> **1. 对每个模型 <span class="math">$M_i$</span>，在 <span class="math">$S$</span> 上进行训练，得到假设 <span class="math">$h_i$</span>**
> 
> **2. 从所有得到的假设 <span class="math">$h_i$</span> 中选择训练误差最小的假设**

为什么这个算法是不可行的呢？考虑选择一个多项式的阶数的情况，多项式的阶数越高，它拟合训练集的效果越好，进而训练误差越低。因此，这种方法总是倾向于选择一个高方差、高阶次的多项式模型，我们之前已经看到这通常是一个较差的选择。

这里给出一个更好的方法：**留出法交叉验证（hold-out cross validation）**，也称为 **简单交叉验证（simple cross validation）**。该算法将按照以下步骤进行：

> [!TIP]
> **1. 随机将 <span class="math">$S$</span> 分成两部分：训练集 <span class="math">$S_{\text{train}}$</span> （大约70%的数据）和验证集 <span class="math">$S_{\text{cv}}$</span> （剩余的30%）。这里 <span class="math">$S_{\text{cv}}$</span> 被称为留出集**
>
> **2. 只在 <span class="math">$S_{\text{train}}$</span> 上训练每个模型 <span class="math">$M_i$</span>，得到假设   <span class="math">$h_i$</span>**
>
> **3. 筛选并输出对保留交叉验证集有最小误差 <span class="math">$\hat{\varepsilon_{S_{cv}}}(h_i)$</span> 的假设 <span class="math">$h_i$</span>（注意，这里 <span class="math">$\hat{\varepsilon_{S_{cv}}}(h)$</span> 表示 <span class="math">$h$</span> 在 <span class="math">$S_{cv}$</span> 上的经验误差）**

通过在模型未训练的示例集 <span class="math">$ S_{cv} $</span> 上测试，我们可以更好地估计每个假设的真实泛化误差，并选择估计泛化误差最小的一个。通常，<span class="math">$1/4 - 1/3$</span> 的数据用作留出验证集，<span class="math">$30\\%$</span> 是一个典型选择。

另外，算法的第三步也可以替换为选择最小化 <span class="math">$ \hat{\varepsilon_{S_{cv}}}(h_i) $</span> 的模型 <span class="math">$ M_i $</span>，然后在整个训练集 <span class="math">$ S $</span> 上重新训练 <span class="math">$ M_i $</span>。这通常能取得一个不错的效果，除非某些学习算法对初始条件和对数据扰动非常敏感，即使 <span class="math">$ M_i $</span> 在 <span class="math">$ S_{\text{train}} $</span> 上表现良好也不一定意味着它在 <span class="math">$ S_{\text{cv}} $</span> 上也会表现得好，可能更好的做法是放弃再训练的步骤。

留出法交叉验证的一个缺点是它"浪费"了约30%的数据。即使我们采取了可选的重新训练步骤。即使我们在整个训练集上训练模型，我们实际上还是在尝试找到一个适合于我们只有 <span class="math">$0.7m$</span> 个训练样本的学习问题的模型，因为我们测试的模型是在每次只使用 <span class="math">$0.7m$</span> 个样本训练的。如果数据充足或者在数据稀缺的学习问题中（例如只有20个样本的问题），我们希望采取更好的方法。

这里介绍另一种方法： **K 折交叉验证（k-fold cross validation）**，它每次使用更少的数据：

> [!TIP]
> **1. 将 <span class="math">$ S $</span> 随机分成 <span class="math">$ k $</span> 个互不相交的子集，每个子集包含 <span class="math">$ m/k $</span> 个训练样本。我们称这些子集为 <span class="math">$ S_1, \ldots, S_k $</span>**
>
> **2. 对每个模型 <span class="math">$ M_i $</span>，按如下方式评估：**
>   - **对于 <span class="math">$ j = 1, \ldots, k $</span>：**
       - **在 <span class="math">$ S_1 \cup \ldots \cup S_{j-1} \cup S_{j+1} \cup \ldots \cup S_k $</span> 上训练模型 <span class="math">$ M_i $</span>，即在除了 <span class="math">$ S_j $</span> 之外的所有数据上训练，得到假设 <span class="math">$ h_{ij} $</span>，然后在 <span class="math">$ S_j $</span>上测试 <span class="math">$ h_{ij} $</span>，得到 <span class="math">$ \hat{\varepsilon_{S_j}}(h_{ij}) $</span>**
       - **对 <span class="math">$ \hat{\varepsilon_{S_j}}(h_{ij}) $</span> 取平均值即得到 <span class="math">$ M_i $</span> 的估计泛化误差**
>
> **3. 选择估算泛化误差最低的模型 <span class="math">$ M_i $</span>，并在整个训练集 <span class="math">$ S $</span> 上重新训练该模型。最后得到我们需要的假设**

通常选择的折数 <span class="math">$ k $</span> 是 10。这样，每次留出的数据量现在是 <span class="math">$ 1/k $</span>，远小于之前的留出交叉验证中的数据量。这种方法可能比留出交叉验证更耗费计算资源，因为我们现在需要对每个模型训练 <span class="math">$ k $</span> 次。

在数据极其稀缺的情况下，有时我们会选择让 <span class="math">$ k = m $</span>。在这种测试中，我们反复在 <span class="math">$ S $</span> 中除了某一个样本外的其他所有样本上进行训练，并在被留出的样本上进行测试。然后将得到的 <span class="math">$ m = k $</span> 个错误取平均值，这样就得到了泛化误差的估算。由于我们每次只留出一个训练样本，这种方法也称为 **留一法交叉验证（leave-one-out cross validation）**。

这些方法不仅可以用于选择模型，也可以用于简单地评估单个模型或算法的性能。例如，它们可以用来更直接地评估单个模型或算法。

### 特征选择

在模型选择中，一个特殊且重要的情况是特征选择。假设你有一个监督学习问题，其中特征数量 <span class="math">$ n $</span> 非常多（可能 <span class="math">$ n \gg m $</span>），但你怀疑只有一部分特征是与学习任务"相关"的。即使对 <span class="math">$ n $</span> 个输入特征，使用一个简单的线性分类器（如感知机），你的假设类的 VC 维仍将是 <span class="math">$ O(n) $</span>，除非训练集非常大，否则过拟合都将是一个潜在问题。

在这样的情况下，我们可以采用一种特征选择算法，来降低特征值的数目。给定 <span class="math">$ n $</span> 个特征，有 <span class="math">$ 2^n $</span> 种可能的特征子集（每个特征都可以包括或不包括在某个特征子集中）。因此，特征选择可以被视为一种模型选择问题：对 <span class="math">$ 2^n $</span> 种可能的模型进行选择。对于 <span class="math">$ n $</span> 值很大的情况，通常开销过大以至于无法显式地枚举和比较所有 <span class="math">$ 2^n $</span> 个模型，因此通常使用某种启发式搜索程序来找到一个好的特征子集。下面的过程被称为 **前向搜索（forward search）**：

> [!TIP]
> 1. 初始化 <span class="math">$ \mathcal{F} = \emptyset $</span>
> 2. 重复以下步骤：
>   - <span class="math">$(a)$</span> 对于 <span class="math">$ i = 1, \ldots, n $</span>，若 <span class="math">$ i \notin \mathcal{F} $</span>，令 <span class="math">$ \mathcal{F}_i = \mathcal{F} \cup \\{i\\} $</span>，并使用某种交叉验证来评估特征 <span class="math">$ \mathcal{F}_i $</span>。（即只使用 <span class="math">$ \mathcal{F}_i $</span> 中的特征训练你的学习算法，并估计其泛化误差。）
>   - <span class="math">$(b)$</span> 将 <span class="math">$ \mathcal{F} $</span> 设置为在步骤 <span class="math">$(a)$</span> 中找到的最佳特征子集。
> 3. 选择并输出在整个搜索过程中筛选出的最佳特征子集。

算法的外循环可以在 <span class="math">$ \mathcal{F} = \\{1, \ldots, n\\} $</span> 达到全部特征规模时停止（即所有特征的集合）或当 <span class="math">$ |\mathcal{F}| $</span> 超过某个预先设定的阈值时终止，这个阈值对应于你希望算法考虑使用的最大特征数。

上述算法描述了 **包装式模型特征选择（wrapper model feature selection）** 的一个实例，此算法本身就是一个将学习算法进行"打包"的过程，反复调用学习算法来评估不同的特征子集的表现。除了前向搜索，还可以使用其他搜索程序。例如，**反向搜索（backward search）** 从集合 <span class="math">$ \mathcal{F} = \\{1, \ldots, n\\} $</span> 开始，一个包含了所有特征的集合，每次删除一个特征，并类似地评估单个特征删除的情况，直到 <span class="math">$ \mathcal{F} = \emptyset $</span>。这种包装式特征选择算法通常工作得很好，但是计算成本高，因为它们需要反复调用学习算法。在给定的包装式算法中（如前向搜索），从空集到 <span class="math">$ \mathcal{F} = \{1, \ldots, n\} $</span> 开始可能需要对学习算法进行 <span class="math">$ O(n^2) $</span> 次调用。

**过滤式特征选择（Filter feature selection）** 提供启发式地选择特征，且计算上更加高效。这些方法通过计算某些简单的分数 <span class="math">$ S(i) $</span> 来评估每个特征的信息量，该分数衡量每个特征 <span class="math">$ x_i $</span> 对类别标签 <span class="math">$ y $</span> 所能体现的信息量。然后，根据需要选择得分最高的 <span class="math">$ k $</span> 个特征。

一个可能的分数选择是将 <span class="math">$ S(i) $</span> 定义为特征 <span class="math">$ x_i $</span> 与 <span class="math">$ y $</span> 之间的相关系数（或其绝对值），从而选择与类别标签最强相关的特征。实际上，通常（尤其是当 <span class="math">$ x_i $</span> 为离散值时）选择 <span class="math">$ x_i $</span> 和 <span class="math">$ y $</span> 之间的 **互信息（mutual information）**，记作 <span class="math">$ \text{MI}(x_i,y) $</span>，来作为 <span class="math">$ S(i) $</span>。互信息可以如下计算：

<div class="math">
$$
\text{MI}(x_i, y) = \sum_{x_i \in \{0,1\}} \sum_{y \in \{0,1\}} p(x_i, y) \log \left( \frac{p(x_i, y)}{p(x_i)p(y)} \right)
$$
</div>

这个公式假设 <span class="math">$ x_i $</span> 和 <span class="math">$ y $</span> 是二值的；对于更一般的情况，会超过变量的所有取值域。概率项 <span class="math">$ p(x_i, y) $</span>, <span class="math">$ p(x_i) $</span>, 和 <span class="math">$ p(y) $</span> 可以根据它们在训练集上的经验分布来估计。

为了直观理解这个分数的作用，互信息也可以表示为 **KL（Kullback-Leibler）散度**：

<div class="math">
$$
\text{MI}(x_i, y) = \text{KL}(p(x_i, y) || p(x_i)p(y))
$$
</div>

KL 散度提供了一个衡量两个概率分布差异的度量。如果 <span class="math">$x_i$</span> 和 <span class="math">$y$</span> 是独立的随机变量，那么必然有 <span class="math">$p(x_i, y) = p(x_i)p(y)$</span>，而两个分布之间的 KL 散度将为零。这与 <span class="math">$x_i$</span> 和 <span class="math">$y$</span> 独立时 <span class="math">$x_i$</span> 对于 <span class="math">$y$</span> 所持有的信息量很少的观点一致，那么 <span class="math">$S(i)$</span> 的分数理所应当很小。相反，如果 <span class="math">$x_i$</span> 对 <span class="math">$y$</span> 来说所持有的信息量很大，那么它们的互信息值 <span class="math">$MI(x_i, y)$</span> 也将会变得很大。

最后一个细节：现在你已经根据特征的分数 <span class="math">$S(i)$</span> 对特征进行了排序，你如何确定选择特征数 <span class="math">$k$</span> 呢？一个标准方法是使用交叉验证来在可能的 <span class="math">$k$</span> 值中选择。例如，对文本分类使用朴素贝叶斯方法，词汇规模 <span class="math">$n$</span> 通常非常大，那么使用这种方法选择特征子集通常会提高分类器的准确性。

### 贝叶斯统计与正则化*

在本节中，我们将讨论另一个用于防止过拟合的方法。

在开始时，我们讨论了使用最大似然估计（ML）进行参数拟合，并根据以下方式选择了我们的参数：

<div class="math">
$$
\theta_{\text{ML}} = \arg\max_\theta \prod_{i=1}^m p(y^{(i)} | x^{(i)}; \theta)
$$
</div>

在我们后续的讨论中，我们将 <span class="math">$\theta$</span> 视为一个未知的参数。但在 **频率统计（frequentist statistics）** 中，往往认为 <span class="math">$\theta$</span> 是一个未知的常量。在频率学派的世界观中，<span class="math">$\theta$</span> 只是碰巧未知，而不是随机的。而我们的任务就是要找出某种统计过程例如最大似然，来对这些参数进行估计。

另一种处理我们的参数估计问题的方法是采用贝叶斯世界观，将 <span class="math">$\theta$</span> 视为一个随机变量，其值未知。在这种方法中，我们会指定一个关于 <span class="math">$\theta$</span> 的 **先验分布（prior distribution）** <span class="math">$p(\theta)$</span>，表达我们对参数的"先验信念程度"。给定一个训练集 <span class="math">$S = \\{(x^{(i)}, y^{(i)})\\}_{i=1}^m$</span>，当我们被要求对一个新的 <span class="math">$x$</span> 值做出预测时，我们可以计算在参数上的后验分布：

<div class="math">
$$
\begin{align*}
p(\theta | S) &= \frac{p(S | \theta) p(\theta)}{p(S)} \\[5pt]
&= \frac{\left( \prod_{i=1}^{m} p(y^{(i)} | x^{(i)}, \theta) \right) p(\theta)}{\int_{\theta} \left( \prod_{i=1}^{m} p(y^{(i)} | x^{(i)}, \theta) \right) p(\theta) d\theta} \tag{1}
\end{align*}
$$
</div>

在上面的公式中，<span class="math">$p(y^{(i)}|x^{(i)}, \theta)$</span> 来自于你所使用的学习模型。例如，如果你使用的是贝叶斯逻辑回归，你可能选择 <span class="math">$p(y^{(i)}|x^{(i)}, \theta) = h_{\theta}(x^{(i)})^{y^{(i)}} (1 - h_{\theta}(x^{(i)}))^{(1-y^{(i)})}$</span>，其中 <span class="math">$\displaystyle{h_{\theta}(x^{(i)}) = \frac{1}{1 + \exp(-\theta^T x^{(i)})}}$</span>。

> [!NOTE]
> 由于我们在这里把 <span class="math">$\theta$</span> 看作是一个随机变量了，就完全可以在其值上使用条件判断。也就是使用
<span class="math">$p(y|x,\theta)$</span> 而非 <span class="math">$p(y|x;\theta)$</span>

当我们获得一个新的测试样例 <span class="math">$x$</span> 并对其进行预测时，我们可以使用类别标签的后验分布来计算参数 <span class="math">$\theta$</span> 的后验分布：

<div class="math">
$$
p(y|x, S) = \int_\theta p(y|x, \theta) p(\theta|S) d\theta \tag{2}
$$
</div>

> [!NOTE]
> 如果 <span class="math">$y$</span> 是一个离散值，那么此处的积分可以用求和来代替

这里的 <span class="math">$p(\theta|S)$</span> 来自于公式<span class="math">$(1)$</span>。例如，如果目标是预测给定 <span class="math">$x$</span> 条件下的 <span class="math">$y$</span> 的期望值，那么我们会输出：

<div class="math">
$$
E[y|x, S] = \int_y y p(y|x, S) dy
$$
</div>

我们在这里概述的过程可以被认为是"完全贝叶斯"预测。我们的预测是通过取关于 <span class="math">$\theta$</span> 的后验分布 <span class="math">$p(\theta|S)$</span> 的平均值来计算的。不幸的是，通常很难计算这种后验分布，因为它需要对（通常是高维的）<span class="math">$\theta$</span> 进行积分，这通常不能以闭合形式完成。

因此，我们通常会采用一种近似方法来近似后验分布 <span class="math">$\theta$</span>。一个常见的近似是用单点估计来替代后验分布，这称为 **最大后验估计 MAP（maximum a posteriori）**：

<div class="math">
$$
\theta_{\text{MAP}} = \arg \max_{\theta} \prod_{i=1}^m p(y^{(i)}|x^{(i)}, \theta) p(\theta) \tag{3}
$$
</div>

注意，实际上其与最大似然估计（ML）的公式相同，只是在最后加上了先验项 <span class="math">$p(\theta)$</span>。

在实际应用中，先验 <span class="math">$p(\theta)$</span> 的常见选择是假设 <span class="math">$\theta \sim N(0, \tau^2 I)$</span>。使用这样的一个先验概率分布，拟合的参数 <span class="math">$\theta_{MAP}$</span> 将具有比最大似然选出的参数有着更小的范数。实际上，这使得贝叶斯 MAP 估计对防止过拟合的能力强于 ML 估计。例如，在文本分类中，即使我们通常有 <span class="math">$n \gg m$</span>，而贝叶斯逻辑回归也被证明是一种有效的算法。
