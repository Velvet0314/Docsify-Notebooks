# 第十三章&ensp;强化学习

在这一章中，我们开始学习强化学习和自适应控制。在监督学习中，我们看到算法尝试使它们的输出模仿训练集中的标签 <span class="math">$ y $</span>。在这种情况下，标签为每个输入 <span class="math">$ x $</span> 提供了明确的“正确答案”。相比之下，对于许多序列决策和控制问题，很难为学习算法提供这种明确的监督。例如，当我们刚刚制造了一台四足机器人并尝试编程让它行走时，起初我们完全不知道“正确”的动作是什么，因此也不知道如何为学习算法提供明确的监督来模仿。

在强化学习框架中，我们会给算法提供一个奖励函数，该函数向学习代理指示它何时表现良好，何时表现不佳。

强化学习已经成功应用于许多领域，如自主直升机飞行、手机网络路由、市场策略选择、工厂控制以及高效的网页索引。我们对强化学习的研究将从 **马尔可夫决策过程 (Markov decision processes)** 的定义开始，该定义为强化学习问题的形式化提供了基础。

通过下面的学习，应该重点掌握：

* 马尔可夫决策过程

- - -

### 马尔可夫决策过程

### 1 马尔可夫决策过程 (Markov Decision Processes)

一个马尔可夫决策过程 (MDP) 是一个五元组 <span class="math">$ (S, A, \{P_{sa}\}, \gamma, R) $</span>，其中：

- **S** 是状态集。例如，在自主直升机飞行中， <span class="math">$ S $</span> 可能是直升机所有可能的位置和姿态的集合。
- **A** 是动作集。例如，控制直升机的操作杆可以推动的所有可能方向的集合。
- <span class="math">$ P_{sa} $</span> 是状态转移概率。对于每个状态 <span class="math">$ s \in S $</span> 和动作 <span class="math">$ a \in A $</span>，<span class="math">$ P_{sa} $</span> 是一个定义在状态空间上的分布。稍后我们会更详细地讨论这一点，简单来说，<span class="math">$ P_{sa} $</span> 给出了我们在状态 <span class="math">$ s $</span> 执行动作 <span class="math">$ a $</span> 后转移到的状态的概率分布。
- <span class="math">$ \gamma \in [0, 1] $</span> 被称为折扣因子 (discount factor)。
- <span class="math">$ R : S \times A \rightarrow \mathbb{R} $</span> 是奖励函数 (reward function)。奖励有时也可以表示为仅状态的函数，此时我们有 <span class="math">$ R : S \rightarrow \mathbb{R} $</span>。

MDP 的动态过程如下：我们从某个状态 <span class="math">$ s_0 $</span> 开始，并选择一个动作 <span class="math">$ a_0 \in A $</span> 来执行。由于我们的选择，MDP 的状态随机地转移到某个后继状态 <span class="math">$ s_1 $</span>，该状态由 <span class="math">$ s_1 \sim P_{s_0 a_0} $</span> 给出。然后，我们可以选择另一个动作 <span class="math">$ a_1 $</span>。由于这个动作，状态再次转移，现在是 <span class="math">$ s_2 \sim P_{s_1 a_1} $</span>。然后我们选择 <span class="math">$ a_2 $</span>，依此类推。图示上，我们可以将这个过程表示如下：

<div class="math">
$$
s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} s_3 \xrightarrow{a_3} \dots
$$
</div>

当访问状态序列 <span class="math">$ s_0, s_1, s_2, \dots $</span> 并执行动作 <span class="math">$ a_0, a_1, \dots $</span> 时，我们的总回报 (total payoff) 由以下公式给出：

<div class="math">
$$
R(s_0, a_0) + \gamma R(s_1, a_1) + \gamma^2 R(s_2, a_2) + \dots
$$
</div>

或者，当我们将奖励仅作为状态的函数来编写时，公式变为：

<div class="math">
$$
R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots
$$
</div>

在我们大部分的推导过程中，我们将使用更为简单的状态奖励函数 <span class="math">$ R(s) $</span>，尽管将其推广为状态-动作奖励函数 <span class="math">$ R(s, a) $</span> 并不会带来特别的困难。

在强化学习中，我们的目标是选择一系列动作，以最大化总回报的期望值：

<div class="math">
$$
\mathbb{E} \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots \right]
$$
</div>

注意，在时间步 <span class="math">$ t $</span> 的奖励会被折扣因子 <span class="math">$ \gamma^t $</span> 折扣。因此，为了使期望回报最大化，我们希望尽可能早地获得正向奖励，并尽量推迟负向奖励。在经济学应用中，假设 <span class="math">$ R(\cdot) $</span> 代表获得的金钱数量，折扣因子 <span class="math">$ \gamma $</span> 也可以用利率解释（今天的一美元比明天的一美元更有价值）。

---

**策略** (Policy) 是一个函数 <span class="math">$ \pi : S \mapsto A $</span>，它将状态映射到动作。我们说我们正在执行某个策略 <span class="math">$ \pi $</span> 时，意味着当我们处于状态 <span class="math">$ s $</span> 时，我们选择动作 <span class="math">$ a = \pi(s) $</span>。我们也定义了该策略 <span class="math">$ \pi $</span> 的 **价值函数** (Value Function) <span class="math">$ V^\pi $</span>，其定义为：

<div class="math">
$$
V^\pi(s) = \mathbb{E} \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots \mid s_0 = s, \pi \right]
$$
</div>

<span class="math">$ V^\pi(s) $</span> 是从状态 <span class="math">$ s $</span> 开始并按照策略 <span class="math">$ \pi $</span> 执行动作时，未来折扣奖励的期望总和。

给定一个固定的策略 <span class="math">$ \pi $</span>，它的价值函数 <span class="math">$ V^\pi $</span> 满足 **贝尔曼方程** (Bellman Equation)：

<div class="math">
$$
V^\pi(s) = R(s) + \gamma \sum_{s' \in S} P_{s\pi(s)}(s') V^\pi(s')
$$
</div>

这表明，状态 <span class="math">$ s $</span> 开始的期望折扣奖励总和由两个部分组成：第一部分是立即奖励 <span class="math">$ R(s) $</span>，它仅因为我们处于状态 <span class="math">$ s $</span> 而获得；第二部分是未来折扣奖励的期望值。更详细地观察第二部分，我们可以将上式中的求和项重写为 <span class="math">$ \mathbb{E}_{s' \sim P_{s\pi(s)}}[V^\pi(s')] $</span>。这意味着第二部分是状态 <span class="math">$ s' $</span> 开始的折扣奖励总和的期望值，其中状态 <span class="math">$ s' $</span> 根据 <span class="math">$ P_{s\pi(s)} $</span> 分布，这是在状态 <span class="math">$ s $</span> 执行动作 <span class="math">$ \pi(s) $</span> 后可能转移到的状态的分布。因此，这个第二部分给出了在 MDP 中经过第一步后的折扣奖励总和的期望。

贝尔曼方程可以有效地用于求解 <span class="math">$ V^\pi $</span>。具体来说，在有限状态的 MDP 中（即 <span class="math">$ |S| < \infty $</span>），我们可以为每个状态 <span class="math">$ s $</span> 写出这样一个方程。这将给出一个包含 <span class="math">$ |S| $</span> 个未知变量（每个状态对应一个 <span class="math">$ V^\pi(s) $</span>）的线性方程组，这个方程组可以高效地解出每个 <span class="math">$ V^\pi(s) $</span>。

### 最优价值函数与最优策略

我们还定义了 **最优价值函数** (Optimal Value Function)，其定义为：

<div class="math">
$$
V^*(s) = \max_\pi V^\pi(s) \tag{1}
$$
</div>

换句话说，这是通过任何策略可以达到的、期望折扣回报总和的最优值。关于最优价值函数的贝尔曼方程也有一个版本：

<div class="math">
$$
V^*(s) = R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^*(s') \tag{2}
$$
</div>

上式中的第一项仍然是立即奖励 <span class="math">$ R(s) $</span>。第二项是对所有动作 <span class="math">$ a \in A $</span> 来说，执行动作 <span class="math">$ a $</span> 后未来折扣奖励期望总和的最大值。你应该确保理解这个方程，并明白它的合理性。

我们还定义了一个策略 <span class="math">$ \pi^* : S \mapsto A $</span>，其定义如下：

<div class="math">
$$
\pi^*(s) = \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s') V^*(s') \tag{3}
$$
</div>

注意，<span class="math">$ \pi^*(s) $</span> 给出了在方程 (2) 中取得最大值的动作 <span class="math">$ a $</span>。

这是一个事实，对于每个状态 <span class="math">$ s $</span> 和任意策略 <span class="math">$ \pi $</span>，我们有：

<div class="math">
$$
V^*(s) = V^{\pi^*}(s) \geq V^\pi(s)
$$
</div>

第一个等式表明，<span class="math">$ V^{\pi^*} $</span>（即对应于最优策略 <span class="math">$ \pi^* $</span> 的价值函数）等于对于所有状态 <span class="math">$ s $</span> 的最优价值函数 <span class="math">$ V^*(s) $</span>。进一步，不等式表明，最优策略 <span class="math">$ \pi^* $</span> 的价值至少和其他任何策略的价值一样大。换句话说，按照方程 (3) 定义的 <span class="math">$ \pi^* $</span> 是最优策略。

请注意，<span class="math">$ \pi^* $</span> 有一个有趣的性质：它对于所有状态 <span class="math">$ s $</span> 都是最优的策略。具体来说，情况并不是这样的：如果我们从某个状态 <span class="math">$ s $</span> 开始，就有某个针对该状态的最优策略，而如果我们从另一个状态 <span class="math">$ s' $</span> 开始，就有另一个针对 <span class="math">$ s' $</span> 的最优策略。实际上，相同的策略 <span class="math">$ \pi^* $</span> 在所有状态下都能在方程 (1) 中取得最大值。这意味着，无论我们从哪个初始状态开始，我们都可以使用相同的最优策略 <span class="math">$ \pi^* $</span> 来解决我们的马尔可夫决策过程 (MDP)。

### 值迭代和策略迭代

我们现在描述两种高效算法来解决有限状态的马尔可夫决策过程 (MDP) 问题。此处，我们仅讨论状态和动作空间都是有限的 MDP（即 <span class="math">$ |S| < \infty, |A| < \infty $</span>）。

第一种算法，**价值迭代** (Value Iteration)，过程如下：

1. 对每个状态 <span class="math">$ s $</span>，初始化 <span class="math">$ V(s) := 0 $</span>。
2. 重复直到收敛 {
   对每个状态，更新 <span class="math">$ V(s) := R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V(s') $</span>
}

该算法可以看作是反复尝试使用贝尔曼方程 (2) 来更新估计的价值函数。内部循环中有两种更新方式。第一种方式是，我们可以先为每个状态 <span class="math">$ s $</span> 计算新的 <span class="math">$ V(s) $</span> 值，然后用这些新值覆盖所有旧值。这称为 **同步更新** (Synchronous Update)。在这种情况下，该算法可以看作是实现了一个 “贝尔曼备份操作符”(Bellman backup operator)，它将当前的价值函数映射到一个新的估计值。（详见作业问题。）另一种方式是进行 **异步更新** (Asynchronous Update)，这里我们会按某种顺序循环遍历状态，逐个更新它们的值。

无论是同步还是异步更新，都可以证明价值迭代会使 <span class="math">$ V $</span> 收敛到 <span class="math">$ V^* $</span>。找到 <span class="math">$ V^* $</span> 后，我们可以使用方程 (3) 来找到最优策略。

除了价值迭代，找到 MDP 最优策略的另一种标准算法是 **策略迭代** (Policy Iteration)。该算法步骤如下：

1. 随机初始化策略 <span class="math">$ \pi $</span>。
2. 重复直到收敛 {
   (a) 令 <span class="math">$ V := V^\pi $</span>。
   (b) 对每个状态 <span class="math">$ s $</span>，令 <span class="math">$ \pi(s) := \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s') V(s') $</span>
}

因此，内部循环会反复计算当前策略的价值函数，然后使用该价值函数更新策略。在步骤 (b) 中找到的策略 <span class="math">$ \pi $</span> 也被称为 **相对于 <span class="math">$ V $</span> 的贪婪策略** (Greedy with respect to <span class="math">$ V $</span>)。注意，步骤 (a) 可以使用之前描述的贝尔曼方程来完成，在一个固定的策略下，这是一组关于 <span class="math">$ |S| $</span> 个变量的线性方程组。

经过有限次数的迭代，该算法中的价值 <span class="math">$ V $</span> 会收敛到 <span class="math">$ V^* $</span>，策略 <span class="math">$ \pi $</span> 也会收敛到 <span class="math">$ \pi^* $</span>。

价值迭代和策略迭代都是解决马尔可夫决策过程 (MDP) 的标准算法，目前尚未有普遍共识认为哪种算法更优。对于小规模的 MDP，策略迭代通常非常快速，并且能够在很少的迭代次数内收敛。然而，对于具有大状态空间的 MDP，明确求解 <span class="math">$ V^\pi $</span> 可能需要解决一个大型的线性方程组，这可能会很困难。在这些问题中，价值迭代可能是更好的选择。因此，实际上，价值迭代似乎比策略迭代更常被使用。

### MDPs 的模型构建

到目前为止，我们已经讨论了 MDP 及假设已知状态转移概率和奖励的相关算法。在许多现实问题中，我们并不知道状态转移概率和奖励，而是必须从数据中估计它们（通常，<span class="math">$ S $</span>、<span class="math">$ A $</span> 和 <span class="math">$ \gamma $</span> 是已知的）。

例如，假设对于倒立摆问题（见问题集4），我们有多次 MDP 试验，过程如下：

<div class="math">
$$
s_0^{(1)} \xrightarrow{a_0^{(1)}} s_1^{(1)} \xrightarrow{a_1^{(1)}} s_2^{(1)} \xrightarrow{a_2^{(1)}} s_3^{(1)} \dots
$$
</div>
<div class="math">
$$
s_0^{(2)} \xrightarrow{a_0^{(2)}} s_1^{(2)} \xrightarrow{a_1^{(2)}} s_2^{(2)} \xrightarrow{a_2^{(2)}} s_3^{(2)} \dots
$$
</div>

其中，<span class="math">$ s_i^{(j)} $</span> 是在第 <span class="math">$ j $</span> 次试验的第 <span class="math">$ i $</span> 个时间步时的状态，<span class="math">$ a_i^{(j)} $</span> 是在该时间步采取的对应动作。每次试验可能运行直到 MDP 终止（例如，倒立摆跌倒），或者可能在某些较大的但有限的时间步数后停止。

基于这些在 MDP 中的“经验”，我们可以很容易地通过最大似然估计推导出状态转移概率的估计：

<div class="math">
$$
P_{sa}(s') = \frac{\#\text{在状态 } s \text{ 下采取动作 } a \text{ 并转移到 } s' \text{ 的次数}}{\#\text{在状态 } s \text{ 下采取动作 } a \text{ 的次数}} \tag{4}
$$
</div>

或者，如果分子和分母均为“0/0”——即从未在状态 <span class="math">$ s $</span> 下采取过动作 <span class="math">$ a $</span>，我们可以简单地将 <span class="math">$ P_{sa}(s') $</span> 估计为 <span class="math">$ 1/|S| $</span>（即，估计 <span class="math">$ P_{sa} $</span> 为所有状态的均匀分布）。

注意，如果我们获得了更多的经验（即更多的试验），我们可以通过有效的方式更新我们估计的状态转移概率。具体而言，如果我们对分子和分母的计数保持累积，那么随着我们观察到更多的试验，我们可以简单地继续累积这些计数。计算这些计数的比率后即可得到我们新的 <span class="math">$ P_{sa} $</span> 估计。

使用类似的方法，如果 <span class="math">$ R $</span> 是未知的，我们也可以将状态 <span class="math">$ s $</span> 下的即时奖励 <span class="math">$ R(s) $</span> 估计为我们在状态 <span class="math">$ s $</span> 观察到的平均奖励。

在为 MDP 学习到一个模型之后，我们可以使用价值迭代或策略迭代来通过估计的转移概率和奖励解决 MDP 问题。例如，结合模型学习与价值迭代，以下是一个在状态转移概率未知的 MDP 中的学习算法：

1. 随机初始化策略 <span class="math">$ \pi $</span>。
2. 重复 {
   (a) 在 MDP 中执行策略 <span class="math">$ \pi $</span> 进行若干次试验。
   (b) 使用 MDP 中的累积经验，更新 <span class="math">$ P_{sa} $</span>（及 <span class="math">$ R $</span>，如果适用）。
   (c) 使用估计的状态转移概率和奖励执行价值迭代，获得新的估计价值函数 <span class="math">$ V $</span>。
   (d) 更新 <span class="math">$ \pi $</span> 为相对于 <span class="math">$ V $</span> 的贪婪策略。
}

我们注意到，对于该特定算法，有一个简单的优化可以使其运行更快。具体而言，在算法的内部循环中，当我们应用价值迭代时，如果我们不将价值迭代初始化为 <span class="math">$ V = 0 $</span>，而是用上一轮迭代中的解作为初始化状态，这将为价值迭代提供更好的初始值，并使其更快速地收敛。

### 连续状态的 MDPs

到目前为止，我们的讨论集中在具有有限状态数目的 MDP 上。现在，我们讨论可能具有无限状态数目的 MDP 算法。例如，对于一辆汽车，我们可以用状态 <span class="math">$ (x, y, \theta, \dot{x}, \dot{y}, \dot{\theta}) $</span> 来表示它的状态，这些状态包括其位置 <span class="math">$ (x, y) $</span>、方向 <span class="math">$ \theta $</span>、沿 <span class="math">$ x $</span> 和 <span class="math">$ y $</span> 方向的速度 <span class="math">$ \dot{x} $</span> 和 <span class="math">$ \dot{y} $</span>，以及角速度 <span class="math">$ \dot{\theta} $</span>。因此，状态空间 <span class="math">$ S = \mathbb{R}^6 $</span> 是一个无限集合。

因为对于汽车来说，位置和方向的可能组合是无限的。同样，你在 PS4 上看到的倒立摆问题的状态是 <span class="math">$ (x, \theta, \dot{x}, \dot{\theta}) $</span>，其中 <span class="math">$ \theta $</span> 是摆杆的角度。再例如，直升机在三维空间飞行的状态可以表示为 <span class="math">$ (x, y, z, \phi, \theta, \psi, \dot{x}, \dot{y}, \dot{z}, \dot{\phi}, \dot{\theta}, \dot{\psi}) $</span>，其中 <span class="math">$ \phi $</span>、<span class="math">$ \theta $</span>、<span class="math">$ \psi $</span> 分别表示直升机的滚转角、俯仰角和偏航角，它们定义了直升机的三维姿态。

在本节中，我们将研究状态空间为 <span class="math">$ S = \mathbb{R}^n $</span> 的场景，并介绍求解此类 MDP 的方法。

#### 离散化

或许解决连续状态 MDP 最简单的方法是对状态空间进行离散化，然后使用诸如价值迭代或策略迭代的算法来求解。如前所述。例如，如果我们有二维状态 <span class="math">$ (s_1, s_2) $</span>，我们可以使用网格来离散化状态空间：

在这里，每个网格单元代表一个离散状态 <span class="math">$ s $</span>。然后我们可以通过一个离散状态 MDP <span class="math">$ (S, A, \{P_{sa}\}, \gamma, R) $</span> 来近似连续状态 MDP，其中 <span class="math">$ S $</span> 是离散的状态集合，<span class="math">$ P_{sa} $</span> 是状态转移概率。我们可以使用价值迭代或策略迭代来对离散状态 MDP 中的 <span class="math">$ V(s) $</span> 和 <span class="math">$ \pi(s) $</span> 进行求解，进而近似连续状态 MDP 的解。

当我们的实际系统处于某个连续状态 <span class="math">$ s \in S $</span> 且需要选择一个动作时，我们可以计算出该连续状态对应的离散状态 <span class="math">$ \hat{s} $</span>，然后执行动作 <span class="math">$ \pi(\hat{s}) $</span>。

这种离散化方法在许多问题中都能起作用，然而它有两个缺点。首先，它对 <span class="math">$ V^* $</span>（以及 <span class="math">$ \pi^* $</span>）的表示较为粗糙。具体来说，它假设价值函数在每个离散区间（即网格单元）上是常值的。

为了更好地理解这种表示的局限性，考虑一个监督学习问题：我们尝试对一个数据集拟合一个函数。显然，线性回归在这个问题中可以很好地工作。然而，如果我们对 x 轴进行离散化，并使用每个离散区间上的分段常数函数来表示数据，那么我们拟合的结果将会表现得像阶梯一样。

这种分段常数表示对于很多平滑函数而言并不是一种好的表示。它会使输入在不同的区间之间显得不连续，并且在不同网格单元之间没有泛化能力。使用这种表示方式，我们需要非常细的离散化（即非常小的网格单元）才能得到一个良好的近似。

这种表示的第二个缺点被称为**维度灾难**（Curse of Dimensionality）。假设 <span class="math">$ S = \mathbb{R}^n $</span>，并且我们将状态空间中的每个维度离散化为 <span class="math">$ k $</span> 个值，那么总的离散状态数就是 <span class="math">$ k^n $</span>。随着状态空间维度 <span class="math">$ n $</span> 的增加，离散状态数会以指数级增长，因此这种方法并不适用于大型问题。例如，对于一个 10 维状态空间，如果我们将每个状态变量离散化为 100 个值，那么我们将有 <span class="math">$ 100^{10} = 10^{20} $</span> 个离散状态，这远远超过现代台式机的处理能力。

作为经验法则，离散化通常在一维和二维问题上效果很好（并且有简单且快速实现的优势）。通过一些巧妙的设计和对离散化方法的仔细选择，它在高达四维的状态空间问题上也往往能取得不错的效果。如果你非常聪明并且有一点运气，你甚至可能在六维问题上让它发挥作用。但对于更高维度的问题，这种方法很少奏效。

#### 值函数近似

我们现在描述另一种方法，用于在连续状态的 MDP 中找到策略。在这种方法中，我们直接对 <span class="math">$ V^* $</span> 进行近似，而不使用离散化。这种方法被称为**价值函数近似**，它已经成功地应用于许多强化学习问题。

##### 使用一个模型或模拟器

为了开发价值函数近似算法，我们假设我们有一个 MDP 的模型或模拟器。非正式地说，模拟器是一个黑箱，它接受任何（连续值的）状态 <span class="math">$ s_t $</span> 和动作 <span class="math">$ a_t $</span> 作为输入，并根据状态转移概率 <span class="math">$ P_{sa} $</span> 输出下一个状态 <span class="math">$ s_{t+1} $</span> 的采样值。

有几种方法可以得到这样的模型。一个方法是使用物理仿真。例如，PS4 上倒立摆问题的模拟器是通过使用物理定律来计算在时间 <span class="math">$ t+1 $</span> 时小车/摆杆的位置和姿态，这取决于当前时刻 <span class="math">$ t $</span> 的状态以及采取的动作 <span class="math">$ a $</span>，前提是我们知道系统的所有参数（如摆杆的长度、摆杆的质量等）。或者，也可以使用现成的物理仿真软件包，该软件包接受机械系统的完整物理描述、当前状态 <span class="math">$ s_t $</span> 和动作 <span class="math">$ a_t $</span> 作为输入，并在系统中计算 <span class="math">$ s_{t+1} $</span>，通常以秒为单位的小时间步进行预测。

另一种获得模型的方法是从 MDP 中收集的数据中进行学习。例如，假设我们进行了 <span class="math">$ m $</span> 次试验，在每次试验中，我们在 MDP 中反复执行动作，每次试验运行 <span class="math">$ T $</span> 个时间步。这可以通过随机选择动作、执行某个特定策略或其他方式选择动作来完成。我们将观察到如下状态序列：

<div class="math">
$$
s_0^{(1)} \xrightarrow{a_0^{(1)}} s_1^{(1)} \xrightarrow{a_1^{(1)}} s_2^{(1)} \dots \xrightarrow{a_{T-1}^{(1)}} s_T^{(1)}
$$
</div>
<div class="math">
$$
s_0^{(2)} \xrightarrow{a_0^{(2)}} s_1^{(2)} \xrightarrow{a_1^{(2)}} s_2^{(2)} \dots \xrightarrow{a_{T-1}^{(2)}} s_T^{(2)}
$$
</div>
<div class="math">
$$
\vdots
$$
</div>
<div class="math">
$$
s_0^{(m)} \xrightarrow{a_0^{(m)}} s_1^{(m)} \xrightarrow{a_1^{(m)}} s_2^{(m)} \dots \xrightarrow{a_{T-1}^{(m)}} s_T^{(m)}
$$
</div>

然后我们可以应用学习算法，将 <span class="math">$ s_{t+1} $</span> 作为 <span class="math">$ s_t $</span> 和 <span class="math">$ a_t $</span> 的函数来预测。例如，可以选择学习以下线性模型：

<div class="math">
$$
s_{t+1} = As_t + Ba_t \tag{5}
$$
</div>

我们可以使用类似于线性回归的算法来估计这个模型。这里，矩阵 <span class="math">$ A $</span> 和 <span class="math">$ B $</span> 是模型的参数，我们可以通过从 <span class="math">$ m $</span> 次试验中收集的数据进行估计，方法是通过最小化以下目标函数来选择 <span class="math">$ A $</span> 和 <span class="math">$ B $</span>：

<div class="math">
$$
\arg \min_{A,B} \sum_{i=1}^m \sum_{t=0}^{T-1} \left\| s_{t+1}^{(i)} - \left( As_t^{(i)} + Ba_t^{(i)} \right) \right\|^2
$$
</div>

这对应于参数的最大似然估计。

在学习到 <span class="math">$ A $</span> 和 <span class="math">$ B $</span> 之后，我们可以构建一个确定性模型，其中 <span class="math">$ s_{t+1} = As_t + Ba_t $</span>，并使用该模型来预测未来的状态。

具体来说，我们始终根据公式 (5) 计算 <span class="math">$ s_{t+1} $</span>。或者，我们还可以构建一个**随机模型** (stochastic model)，在该模型中，<span class="math">$ s_{t+1} $</span> 是输入的一个随机函数，其表示形式为：

<div class="math">
$$
s_{t+1} = A s_t + B a_t + \epsilon_t
$$
</div>

其中 <span class="math">$ \epsilon_t $</span> 是噪声项，通常假设 <span class="math">$ \epsilon_t \sim \mathcal{N}(0, \Sigma) $</span>。（协方差矩阵 <span class="math">$ \Sigma $</span> 也可以直接从数据中估计。）

在这里，我们将下一个状态 <span class="math">$ s_{t+1} $</span> 写成当前状态和动作的线性函数；但当然，也可以使用非线性函数。具体来说，可以学习一个模型 <span class="math">$ s_{t+1} = A_{\phi_s} \phi_s(s_t) + B_{\phi_a} \phi_a(a_t) $</span>，其中 <span class="math">$ \phi_s $</span> 和 <span class="math">$ \phi_a $</span> 是状态和动作的某些非线性特征映射。或者，也可以使用非线性学习算法，例如局部加权线性回归 (locally weighted linear regression)，来学习 <span class="math">$ s_{t+1} $</span> 作为 <span class="math">$ s_t $</span> 和 <span class="math">$ a_t $</span> 的函数的估计。

这些方法都可以用于构建确定性或随机模拟器，用于 MDP 的建模。

##### 拟合值迭代

我们现在描述**拟合价值迭代**算法，该算法用于近似连续状态 MDP 的价值函数。在接下来的内容中，我们假设问题的连续状态空间 <span class="math">$ S = \mathbb{R}^n $</span>，但动作空间 <span class="math">$ A $</span> 是有限且离散的。

回顾价值迭代，我们希望执行以下更新：

<div class="math">
$$
V(s) = R(s) + \gamma \max_{a} \int P_{sa}(s') V(s') ds' \tag{6}
$$
</div>
<div class="math">
$$
V(s) = R(s) + \gamma \max_{a} \mathbb{E}_{s' \sim P_{sa}}[V(s')] \tag{7}
$$
</div>

（在第 2 节中，我们写下了价值迭代方程，其中使用的是求和 <span class="math">$ V(s) := R(s) + \gamma \max_a \sum_{s'} P_{sa}(s') V(s') $</span>，而不是积分。新的表示法反映了我们现在处理的是连续状态而非离散状态。）

拟合价值迭代的主要思想是：我们将对有限采样的状态 <span class="math">$ s^{(1)}, \dots, s^{(m)} $</span> 近似执行上述步骤。具体而言，我们将使用监督学习算法——线性回归或如下所述的方法——来近似将价值函数表示为状态的线性或非线性函数：

<div class="math">
$$
V(s) = \theta^T \phi(s)
$$
</div>

这里，<span class="math">$ \phi $</span> 是某种适当的状态特征映射。

对于我们的有限状态样本集合中的每个状态 <span class="math">$ s $</span>，拟合价值迭代首先计算一个量 <span class="math">$ y(s) $</span>，这是我们对

<div class="math">
$$
R(s) + \gamma \max_a \mathbb{E}_{s' \sim P_{sa}}[V(s')]
$$
</div>

的近似（即方程 (7) 右侧）。然后，它将应用监督学习算法，尝试使 <span class="math">$ V(s) $</span> 接近 

<div class="math">
$$
R(s) + \gamma \max_a \mathbb{E}_{s' \sim P_{sa}}
$$
</div>

（或换句话说，尝试使 <span class="math">$ V(s) $</span> 接近 <span class="math">$ y(s) $</span>）。

详细地说，算法如下：

1. 随机采样 <span class="math">$ m $</span> 个状态 <span class="math">$ s^{(1)}, s^{(2)}, \dots, s^{(m)} \in S $</span>。
2. 初始化 <span class="math">$ \theta = 0 $</span>。
3. 重复 {
   对于 <span class="math">$ i = 1, \dots, m $</span>：
   
   对于每个动作 <span class="math">$ a \in A $</span>：
   
   - 采样 <span class="math">$ s_{t+1}^{(i)} \sim P_{s^i a} $</span>（使用 MDP 模型）。
   - 计算：
   
   <div class="math">
$$
   q(s^i, a) = R(s^i) + \gamma \frac{1}{K} \sum_{k=1}^{K} V(s_{t+1}^{(i, k)})
   $$
</div>
   （因此，<span class="math">$ q(s, a) $</span> 是 <span class="math">$ R(s) + \gamma \mathbb{E}_{s' \sim P_{sa}}[V(s')] $</span> 的估计值。）
   
   设 <span class="math">$ y(s^i) = \max_a q(s^i, a) $</span>。
   
   }
   
   // 在原始价值迭代算法（离散状态）中，我们更新价值函数 <span class="math">$ V(s) $</span> 到 <span class="math">$ V(s) = y(s) $</span>。
   
   // 在此算法中，我们希望 <span class="math">$ V(s) \approx y(s) $</span>，这将通过应用监督学习（例如线性回归）实现。
   
   设 <span class="math">$ \theta = \arg \min_\theta \frac{1}{2} \sum_{i=1}^{m} (V(s^{(i)}) - y(s^{(i)}))^2 $</span>。
   
}

在上文中，我们使用线性回归编写了拟合价值迭代的算法，目的是使 <span class="math">$ V(s^{(i)}) $</span> 接近 <span class="math">$ y(s^{(i)}) $</span>。该算法步骤与标准监督学习（回归）问题完全类似，其中我们有一个训练集 <span class="math">$ (x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(m)}, y^{(m)}) $</span>，并且希望学习一个从 <span class="math">$ x $</span> 到 <span class="math">$ y $</span> 的映射。唯一的区别是，在这里，状态 <span class="math">$ s $</span> 扮演了 <span class="math">$ x $</span> 的角色。尽管我们的描述使用了线性回归，显然其他回归算法（例如局部加权线性回归）也可以使用。

与离散状态上的价值迭代不同，拟合价值迭代不能被证明总是收敛。然而，在实践中，它经常收敛（或近似收敛），并且对于许多问题都能很好地工作。请注意，如果我们使用的是一个确定性模拟器/模型，那么拟合价值迭代可以通过设置 <span class="math">$ k = 1 $</span> 来简化。这是因为方程 (7) 中的期望是关于确定性分布的，因此只需要一个样本就足够准确地计算该期望。否则，如算法伪代码所示，我们必须采样 <span class="math">$ k $</span> 个样本，并取平均值来近似计算期望（见伪代码中 <span class="math">$ q(s, a) $</span> 的定义）。

最后，拟合价值迭代输出的是 <span class="math">$ V $</span>，这是对 <span class="math">$ V^* $</span> 的一个近似。这一近似值可以明确定义我们的策略 <span class="math">$ \pi $</span>。具体来说，当我们的系统处于某个状态 <span class="math">$ s $</span> 时，我们需要选择一个动作 <span class="math">$ a $</span>，我们希望最大化下式：

<div class="math">
$$
\arg \max_a \mathbb{E}_{s' \sim P_{sa}}[V(s')] \tag{8}
$$
</div>

在拟合价值迭代的内层循环中，我们使用类似的方法来近似这个期望。对于每个动作 <span class="math">$ a $</span>，我们采样 <span class="math">$ s' \sim P_{sa} $</span>，并估计期望。值得注意的是，如果我们的模拟器是下述形式 <span class="math">$ s_{t+1} = f(s_t, a_t) + \epsilon_t $</span>，其中 <span class="math">$ f $</span> 是状态的确定性函数（例如 <span class="math">$ f(s_t, a_t) = A s_t + B a_t $</span>），且 <span class="math">$ \epsilon_t $</span> 是均值为零的高斯噪声，那么在这种情况下，我们可以通过下式选择动作 <span class="math">$ a $</span>：

<div class="math">
$$
\arg \max_a V(f(s, a))
$$
</div>

换句话说，这里我们只是将 <span class="math">$ \epsilon_t = 0 $</span>（即忽略模拟器中的噪声），并设置 <span class="math">$ k = 1 $</span>。等价地，我们可以通过使用方程 (8) 来导出该结果：

<div class="math">
$$
\mathbb{E}_{s' \sim P_{sa}}[V(s')] \approx V(\mathbb{E}_{s' \sim P_{sa}}[s']) \tag{9}
$$
</div>
<div class="math">
$$
= V(f(s, a)) \tag{10}
$$
</div>

其中期望是对随机变量 <span class="math">$ s' \sim P_{sa} $</span> 计算的。只要噪声项 <span class="math">$ \epsilon_t $</span> 较小，这通常会是一个合理的近似。

然而，对于那些不适用于这种近似的情况，必须对所有 <span class="math">$ |A| $</span> 个动作进行采样，以便使用模型计算上述期望值，这在计算上可能非常昂贵。
