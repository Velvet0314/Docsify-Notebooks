# 第一章&ensp;机器学习入门

### 机器学习的定义

关于机器学习，有着诸多的定义。但是其中 Tom Mitchell 于1998年提出的定义是最有趣的：

> <span class="blocks">"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." - Tom Mitchell</span>

如果翻译为中文是这样的：

> <span class="blocks">"如果一个程序通过经验 E 在任务 T 的性能衡量 P 方面有所提高，那么就称该程序从经验 E 中学习。"</span>

简单地讲，机器学习就是通过数据（经验）来改进程序的性能。

### 机器学习的分类

#### 背景知识

<span class="math">$ x^{(i)} :\mathrm{Input\ or\ Features}\ 特征$</span>

<span class="math">$ y^{(i)} :\mathrm{Output\ or\ Target}\ 目标$</span>

<span class="math">$ (x^{(i)},y^{(i)}) :\mathrm{Training\ Example}\ 训练样本$</span>

<span class="math">$ \lbrace (x^{(i)},y^{(i)});i=1,2,...,m \rbrace : \mathrm{Training\ Set}\ 训练集$</span>

<span class="math">$ h:\mathcal{X}\rightarrow\mathcal{Y} :\mathrm{hypothesis}\ 假设函数$</span>

#### 监督学习 (Supervised Learning)

监督学习通过对有标签的数据集进行训练，以学习一个模型，使模型能够对任意给定的输入，都能给出一个适合的输出作为预测。

监督学习一般分为 **回归 (Regression)** 和 **分类 (Classification)** 两种问题。

* 回归 (Regression)：预测值为连续值
* 分类 (Classification)：预测值为离散值

#### 无监督学习 (Unsupervised Learning)

无监督学习通过对无标签的数据集进行训练，以学习一个模型，使模型能够对任意给定的输入，都能给出一个适合的输出作为预测。

无监督学习一般分为 **聚类 (Clustering)** 和 **降维 (Dimensionality Reduction)** 两种问题。

* 聚类 (Clustering)：将相似的数据点归为同一类
* 降维 (Dimensionality Reduction)：将高维的数据点映射到低维的空间中
