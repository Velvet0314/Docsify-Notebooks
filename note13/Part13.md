# 第十三章&ensp;强化学习

在这一章中，我们开始学习强化学习和自适应控制。在监督学习中，我们看到算法尝试使它们的输出试图去模仿训练集中的标签 <span class="math">$ y $</span>。在这种情况下，标签为每个输入 <span class="math">$ x $</span> 提供了明确的"正确答案"。相比之下，对于许多序列决策和控制问题，很难为学习算法提供这种明确的监督。

在强化学习框架中，我们会给算法提供一个奖励函数，该函数指示其何时表现良好，何时表现不佳。随着时间的推移，学习算法就会解决如何选择正确的动作来得到最大的奖励。

强化学习已经成功应用于许多领域，如自主直升机飞行、手机网络路由、市场策略选择、工厂控制以及高效的网页索引。我们对强化学习的研究将从 **马尔可夫决策过程 (Markov decision processes)** 的定义开始，该定义为强化学习问题的形式化提供了基础。

通过下面的学习，应该重点掌握：

* 马尔可夫决策过程

- - -

### 马尔可夫决策过程

一个马尔可夫决策过程 (MDP) 是一个元组 <span class="math">$ (S, A, \\{P_{sa}\\}, \gamma, R) $</span>，其中：

- <span class="math">$S$</span> 是 **状态集（set of states）**。（例如，在自主直升机飞行中， <span class="math">$ S $</span> 可能是直升机所有可能的位置和方向的集合）
- <span class="math">$A$</span> 是 **动作集（set of actions）**。（例如，控制直升机的操作杆可以推动的所有可能方向的集合）
- <span class="math">$ P_{sa} $</span> 是状态转移概率。对于每个状态 <span class="math">$ s \in S $</span> 和动作 <span class="math">$ a \in A $</span>，<span class="math">$ P_{sa} $</span> 是一个定义在状态空间上的分布。稍后我们会更详细地讨论这一点，简单来说，<span class="math">$ P_{sa} $</span> 给出了我们在状态 <span class="math">$ s $</span> 执行动作 <span class="math">$ a $</span> 后转移到的状态的概率分布。
- <span class="math">$ \gamma \in [0, 1) $</span> 被称为 **折扣因子（discount factor）**。
- <span class="math">$ R : S \times A \rightarrow \mathbb{R} $</span> 是 **奖励函数 （reward function）**。（奖励函数有时也可以表示为仅对状态的函数，此时我们有 <span class="math">$ R : S \rightarrow \mathbb{R} $</span>）

MDP 的动态过程如下：我们从某个状态 <span class="math">$ s_0 $</span> 开始，并选择一个动作 <span class="math">$ a_0 \in A $</span> 来执行。由于我们的选择，MDP 的状态随机地转移到某个后继状态 <span class="math">$ s_1 $</span>，该状态由 <span class="math">$ s_1 \sim P_{s_0 a_0} $</span> 给出。然后，我们可以选择另一个动作 <span class="math">$ a_1 $</span>。由于这个动作，状态再次转移，现在是 <span class="math">$ s_2 \sim P_{s_1 a_1} $</span>。然后我们选择 <span class="math">$ a_2 $</span>，依此类推。我们可以将这个过程表示如下：

<div class="math">
$$
s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} s_3 \xrightarrow{a_3} \dots
$$
</div>

当访问状态序列 <span class="math">$ s_0, s_1, s_2, \dots $</span> 并执行动作 <span class="math">$ a_0, a_1, \dots $</span> 时，我们得到的 **总回报 (total payoff)** 由以下公式给出：

<div class="math">
$$
R(s_0, a_0) + \gamma R(s_1, a_1) + \gamma^2 R(s_2, a_2) + \dots
$$
</div>

或者，当我们将奖励函数仅作为与状态相关的函数来描述时，公式变为：

<div class="math">
$$
R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots
$$
</div>

在我们大部分的推导过程中，我们将使用更为简单的状态奖励函数 <span class="math">$ R(s) $</span>，尽管将其推广为状态-动作奖励函数 <span class="math">$ R(s, a) $</span> 并不会特别困难。

在强化学习中，我们的目标是选择一系列动作，以最大化总回报的期望值：

<div class="math">
$$
E \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots \right]
$$
</div>

注意，在时间步长 <span class="math">$ t $</span> 上的奖励函数通过一个参数 <span class="math">$ \gamma^t $</span> 而 **缩减（discounted）**。因此，为了使期望回报最大化，我们希望尽可能早地获得正向奖励，并尽量推迟负面奖励（即惩罚）的出现。在经济学应用中，假设 <span class="math">$ R(\cdot) $</span> 代表获得的金钱数量，<span class="math">$ \gamma $</span> 也就自然地可以用利润率来解释（今天的一美元比明天的一美元更有价值）。

有一种 **策略 （Policy）** 是对于一个任意函数 <span class="math">$ \pi : S \mapsto A $</span>，它将状态映射到动作。我们说我们正在 **执行（execute）** 某个策略 <span class="math">$ \pi $</span> 时，就表示当我们处于状态 <span class="math">$ s $</span> 时，我们选择动作 <span class="math">$ a = \pi(s) $</span>。我们也定义了该策略 <span class="math">$ \pi $</span> 的 **值函数 （Value Function）** <span class="math">$ V^\pi $</span> 为：

<div class="math">
$$
V^\pi(s) = E \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \dots \mid s_0 = s, \pi \right]
$$
</div>

<span class="math">$ V^\pi(s) $</span> 是从状态 <span class="math">$ s $</span> 开始并按照策略 <span class="math">$ \pi $</span> 执行动作时，往后折扣后的奖励的期望总和。

给定一个固定的策略函数 <span class="math">$ \pi $</span>，其值函数 <span class="math">$ V^\pi $</span> 满足 **贝尔曼方程 （Bellman Equations）**：

<div class="math">
$$
V^\pi(s) = R(s) + \gamma \sum_{s' \in S} P_{s\pi(s)}(s') V^\pi(s')
$$
</div>

这表明，状态 <span class="math">$ s $</span> 开始的期望折扣奖励总和由两个部分组成：第一部分是立即奖励 <span class="math">$ R(s) $</span>，它仅因为我们处于状态 <span class="math">$ s $</span> 而获得；第二部分是往后折扣后的奖励的期望值。更详细地观察第二部分，我们可以将上式中的求和项重写为 

<div class="math">
$$
E_{s' \sim P_{s\pi(s)}}[V^\pi(s')]
$$
</div>

这意味着第二部分是状态 <span class="math">$ s' $</span> 开始的折扣奖励总和的期望值，其中状态 <span class="math">$ s' $</span> 根据 <span class="math">$ P_{s\pi(s)} $</span> 分布，这是在状态 <span class="math">$ s $</span> 执行动作 <span class="math">$ \pi(s) $</span> 后可能转移到的状态的分布。因此，这个第二部分给出了在 MDP 中经过第一步后的折扣奖励总和的期望。

贝尔曼方程可以有效地用于求解 <span class="math">$ V^\pi $</span>。具体来说，在有限状态的 MDP 中（即 <span class="math">$ |S| < \infty $</span>），我们可以为每个状态 <span class="math">$ s $</span> 写出这样一个方程。这将给出一个包含 <span class="math">$ |S| $</span> 个未知变量（每个状态对应一个 <span class="math">$ V^\pi(s) $</span>）的线性方程组，这个方程组可以高效地解出每个 <span class="math">$ V^\pi(s) $</span>。

接着，我们还定义了 **最优价值函数（Optimal Value Function）**，其定义为：

<div class="math">
$$
V^*(s) = \max_\pi V^\pi(s) \tag{1}
$$
</div>

换句话说，这是通过任何策略可以达到的、折扣奖励总和期望的最优值。对于最优值函数，也有一个版本的贝尔曼方程：

<div class="math">
$$
V^*(s) = R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^*(s') \tag{2}
$$
</div>

上式中的第一项仍然是立即奖励 <span class="math">$ R(s) $</span>。第二项是对所有动作 <span class="math">$ a \in A $</span> 来说，执行动作 <span class="math">$ a $</span> 后，往后折扣奖励期望总和的最大值。

我们还定义了另一个策略函数 <span class="math">$ \pi^* : S \mapsto A $</span> 如下：

<div class="math">
$$
\pi^*(s) = \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s') V^*(s') \tag{3}
$$
</div>

注意，<span class="math">$ \pi^*(s) $</span> 给出了在方程<span class="math">$(2)$</span>中取得最大值的动作 <span class="math">$ a $</span>。

事实上，对于每个状态 <span class="math">$ s $</span> 和任意策略函数 <span class="math">$ \pi $</span>，我们有：

<div class="math">
$$
V^*(s) = V^{\pi^*}(s) \geq V^\pi(s)
$$
</div>

第一个等式表明，<span class="math">$ V^{\pi^*} $</span>（即对应于最优策略 <span class="math">$ \pi^* $</span> 的值函数）等于对于所有状态 <span class="math">$ s $</span> 的最优值函数 <span class="math">$ V^*(s) $</span>。进一步地，不等式表明，最优策略 <span class="math">$ \pi^* $</span> 的值至少和其他任何策略的值一样大。换句话说，按照方程<span class="math">$(3)$</span>定义的 <span class="math">$ \pi^* $</span> 是最优策略。

注意，策略 <span class="math">$\pi^* $</span> 有一个有趣的性质：它是 **所有状态** <span class="math">$s$</span> 的最优策略。具体来说，并不是说如果我们从某个状态 <span class="math">$s$</span> 开始，就会有该状态的某个最优策略；然后如果我们从另一个状态 <span class="math">$s'$</span> 开始，就会有另一个针对 <span class="math">$s'$</span> 的最优策略。实际上，**相同的策略** <span class="math">$\pi^* $</span> 对 **所有状态** <span class="math">$s $</span> 在方程<span class="math">$(1)$</span> 中都能达到最大值。这意味着无论我们的 MDP 初始状态是什么，我们都可以使用相同的策略 <span class="math">$\pi^* $</span>。

### 值迭代和策略迭代

我们现在描述两种高效算法来解决有限状态的马尔可夫决策过程 (MDP) 问题。此处，我们仅讨论状态和动作空间都是有限的 MDP，即 <span class="math">$ |S| < \infty, |A| < \infty $</span>。

> [!TIP]
> **值迭代（Value Iteration）**：
> <div class="math">
> $$
> \begin{array}{l}
> \text{For each state } s, \text{ initialize } V(s) := 0. \\[5pt]
> \text{Repeat until convergence:} \\[5pt]
> \qquad \text{For every state, update:} \\[5pt]
> \qquad \qquad \qquad V(s) := R(s) + \max_{a \in A} \gamma \sum_{s'} P_a(s')V(s') \\
> \}
> \end{array}
> $$
> </div>


该算法可以看作是反复尝试使用贝尔曼方程<span class="math">$ (2) $</span>来更新估计的值函数。内部循环中有两种更新方式。第一种方式是，我们可以先为每个状态 <span class="math">$ s $</span> 计算新的 <span class="math">$ V(s) $</span> 值，然后用这些新值覆盖所有旧值。这称为 **同步更新 （Synchronous Update）**。在这种情况下，该算法可以看作是实现了一个 **"贝尔曼备份运算符"（Bellman backup operator）**，它将当前的价值函数映射到一个新的估计值。另一种方式是进行 **异步更新 （Asynchronous Update）**，这里我们会按某种顺序循环遍历状态，逐个更新它们的值。

无论是同步还是异步更新，都可以证明值迭代会使 <span class="math">$ V $</span> 收敛到 <span class="math">$ V^* $</span>。找到 <span class="math">$ V^* $</span> 后，我们可以使用方程<span class="math">$ (3) $</span>来找到最优策略。

除了值迭代，找到 MDP 最优策略的另一种标准算法是 **策略迭代（Policy Iteration）**。该算法步骤如下：

> [!TIP]
> <div class="math">
> $$
> \begin{array}{l}
> 1. \text{ Initialize } \pi \text{ randomly.} \\[5pt]
> 2. \text{ Repeat until convergence } \{ \\[5pt]
> \qquad (a) \text{ Let } V := V^\pi. \\[5pt]
> \qquad (b) \text{ For each state } s, \text{ let } \pi(s) := \arg\max_{a \in A} \sum_{s'} P_{sa}(s')V(s'). \\[5pt]
> \}
> \end{array}
> $$
> </div>

因此，内部循环会反复计算当前策略的值函数，然后使用该价值函数更新策略。在步骤 (b) 中找到的策略 <span class="math">$ \pi $</span> 也被称为 **相对于 <span class="math">$ V $</span> 的贪婪策略（Greedy with respect to <span class="math">$ V $</span>）**。注意，步骤 (a) 可以使用之前的贝尔曼方程来完成，在一个固定的策略下，这是一组关于 <span class="math">$ |S| $</span> 个变量的线性方程组。

经过有限次数的迭代，该算法中的 <span class="math">$ V $</span> 会收敛到 <span class="math">$ V^* $</span>，<span class="math">$ \pi $</span> 也会收敛到 <span class="math">$ \pi^* $</span>。

值迭代和策略迭代都是解决马尔可夫决策过程 (MDP) 的标准算法，目前尚未有普遍共识认为哪种算法更优。对于小规模的 MDP，策略迭代通常非常快速，并且能够在很少的迭代次数内收敛。然而，对于具有较大规模状态空间的 MDP，确切求解 <span class="math">$ V^\pi $</span> 可能需要解决一个大型的线性方程组，这可能会很困难。在这些问题中，值迭代可能是更好的选择。因此，实际上值迭代比策略迭代更常被使用。

### MDPs 的模型构建

到目前为止，我们已经讨论了 MDP 及其相关的算法。但是这些都是基于一个假设：状态转移概率和奖励函数是已知的。在许多现实问题中，我们并不知道状态转移概率和奖励函数，而是必须从数据中估计它们（通常，<span class="math">$ S $</span>、<span class="math">$ A $</span> 和 <span class="math">$ \gamma $</span> 是已知的）。

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

其中，<span class="math">$ s_i^{(j)} $</span> 是在第 <span class="math">$ j $</span> 次试验的第 <span class="math">$ i $</span> 个时间步长的状态，<span class="math">$ a_i^{(j)} $</span> 是在该时间步长采取的对应动作。每次试验可能一直运行直到 MDP 终止（例如倒立摆摆杆落下），或者可能在某些较大的但有限的时间步长后停止。

基于这些在 MDP 中的"经验"，我们可以很容易地通过最大似然估计推导出状态转移概率的估计：

<div class="math">
$$
P_{sa}(s') = \frac{\text{在状态 } s \text{ 下采取动作 } a \text{ 并转移到 } s' \text{ 的次数}}{\text{在状态 } s \text{ 下采取动作 } a \text{ 的次数}} \tag{4}
$$
</div>

或者，如果分子和分母均为 <span class="math">$0/0$</span>——即从未在状态 <span class="math">$ s $</span> 下采取过动作 <span class="math">$ a $</span>，我们可以简单地将 <span class="math">$ P_{sa}(s') $</span> 估计为 <span class="math">$ 1/|S| $</span>（即，估计 <span class="math">$ P_{sa} $</span> 为所有状态的均匀分布）。

注意，如果我们获得了更多的经验（即更多的试验，更多的观察），我们可以通过有效的方式更新我们估计的状态转移概率。具体而言，如果我们对等式<span class="math">$ (4) $</span>中分子和分母的计数保持累积，那么随着我们观察到更多的试验，我们可以简单地继续累积这些计数。计算这些计数的比率后即可得到我们新的 <span class="math">$ P_{sa} $</span> 估计。

使用类似的方法，如果 <span class="math">$ R $</span> 是未知的，我们也可以将状态 <span class="math">$ s $</span> 下的期望即时奖励 <span class="math">$ R(s) $</span> 估计为我们在状态 <span class="math">$ s $</span> 观察到的平均奖励。

在为 MDP 学习到一个模型之后，我们可以使用值迭代或策略迭代来通过估计的转移概率和奖励函数解决 MDP 问题。例如，结合模型学习与值迭代，以下是一个在状态转移概率未知的 MDP 中的学习算法：

> [!TIP]
> <div class="math">
> $$
> \begin{array}{l}
> 1. \text{ Initialize } \pi \text{ randomly.} \\[5pt]
> 2. \text{ Repeat } \{ \\[5pt]
> \qquad (a) \text{ Execute } \pi \text{ in the MDP for some number of trials.} \\[5pt]
> \qquad (b) \text{ Using the accumulated experience in the MDP, update our estimates for } P_{sa} \text{ (and } R, \text{ if applicable).} \\[5pt]
> \qquad (c) \text{ Apply value iteration with the estimated state transition probabilities and rewards to get a new} \\[5pt]
> \qquad \quad \ \text{ estimated value function } V. \\[5pt]
> \qquad (d) \text{ Update } \pi \text{ to be the greedy policy with respect to } V. \\[5pt]
> \}
> \end{array}
> $$
> </div>

我们注意到，对于该特定算法，有一个简单的优化可以使其运行得更快。具体而言，在算法的内部循环中，当我们应用值迭代时，如果我们不将值迭代初始化为 <span class="math">$ V = 0 $</span>，而是用上一轮迭代中的解作为初始化状态，这将为值迭代提供更好的初始值，并使其更快速地收敛。

### 连续状态的 MDPs

到目前为止，我们的讨论集中在具有有限状态数量的 MDP 上。现在，我们讨论可能具有无限状态数量的 MDP 算法。例如，对于一辆汽车，我们可以用状态 <span class="math">$ (x, y, \theta, \dot{x}, \dot{y}, \dot{\theta}) $</span> 来表示它的状态，这些状态包括其位置 <span class="math">$ (x, y) $</span>、方向 <span class="math">$ \theta $</span>、沿 <span class="math">$ x $</span> 和 <span class="math">$ y $</span> 方向的速度 <span class="math">$ \dot{x} $</span> 和 <span class="math">$ \dot{y} $</span>，以及角速度 <span class="math">$ \dot{\theta} $</span>。其状态空间 <span class="math">$ S = \mathbb{R}^6 $</span> 是一个无限集合。

因为对于汽车来说，位置和方向的可能组合是无限的。类似地，在问题集 4 上看到的倒立摆问题的状态是 <span class="math">$ (x, \theta, \dot{x}, \dot{\theta}) $</span>，其中 <span class="math">$ \theta $</span> 是摆杆的角度。再例如，直升机在三维空间飞行的状态可以表示为 <span class="math">$ (x, y, z, \phi, \theta, \psi, \dot{x}, \dot{y}, \dot{z}, \dot{\phi}, \dot{\theta}, \dot{\psi}) $</span>，其中 <span class="math">$ \phi $</span>、<span class="math">$ \theta $</span>、<span class="math">$ \psi $</span> 分别表示直升机的滚动角、俯仰角和偏航角，它们定义了直升机的三维运动方向。

在本节中，我们将研究状态空间为 <span class="math">$ S = \mathbb{R}^n $</span> 的情况，并介绍求解此类 MDP 的方法。

#### 离散化

或许解决连续状态 MDP 最简单的方法是对状态空间进行离散化，然后使用诸如值迭代或策略迭代的算法来求解。如前所述。例如，如果我们有二维状态 <span class="math">$ (s_1, s_2) $</span>，我们可以使用网格来离散化状态空间：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/09/22/pAMoIG6.png" data-lightbox="image-13" data-title="Discretization">
  <img src="https://s21.ax1x.com/2024/09/22/pAMoIG6.png" alt="Discretization" style="width:100%;max-width:350px;cursor:pointer">
 </a>
</div>

在这里，每个网格单元代表一个离散状态 <span class="math">$ s $</span>。然后我们可以通过一个离散状态 MDP <span class="math">$ (S, A, \{P_{sa}\}, \gamma, R) $</span> 来近似连续状态 MDP，其中 <span class="math">$ S $</span> 是离散的状态集合，<span class="math">$ P_{sa} $</span> 是状态转移概率。我们可以使用值迭代或策略迭代来对离散状态 MDP 中的 <span class="math">$ V(s) $</span> 和 <span class="math">$ \pi(s) $</span> 进行求解，进而近似连续状态 MDP 的解。

当我们的实际系统处于某个连续状态 <span class="math">$ s \in S $</span> 且需要选择一个动作时，我们可以计算出该连续状态对应的离散状态 <span class="math">$ \hat{s} $</span>，然后执行动作 <span class="math">$ \pi(\hat{s}) $</span>。

这种离散化方法在许多问题中都能起作用，然而它有两个缺点。首先，它对 <span class="math">$ V^* $</span>（以及 <span class="math">$ \pi^* $</span>）的表示较为粗糙。具体来说，它假设值函数在每个离散区间（即网格单元）上是常值的（也就是说，值函数是在每个网格单元中分段的常数）。

为了更好地理解这种表示的局限性，考虑一个监督学习问题：

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/09/22/pAMoOZd.png" data-lightbox="image-13" data-title="dataset">
  <img src="https://s21.ax1x.com/2024/09/22/pAMoOZd.png" alt="dataset" style="width:100%;max-width:350px;cursor:pointer">
 </a>
</div>

我们尝试对一个数据集拟合一个函数。显然，线性回归在这个问题中可以很好地工作。然而，如果我们对 <span class="math">$ x $</span> 轴进行离散化，并使用每个离散区间上的分段常数函数来表示数据，那么我们拟合的结果将会表现得像阶梯一样。

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/09/22/pAMoXdA.png" data-lightbox="image-13" data-title="stairs">
  <img src="https://s21.ax1x.com/2024/09/22/pAMoXdA.png" alt="stairs" style="width:100%;max-width:375px;cursor:pointer">
 </a>
</div>

这种分段常数表示对于很多平滑函数而言并不是一种好的表示。它会使输入在不同的区间之间显得不连续，并且在不同网格单元之间没有泛化能力。使用这种表示方式，我们需要非常细的离散化（即非常小的网格单元）才能得到一个良好的近似。

这种表示的第二个缺点被称为 **维度灾难（Curse of Dimensionality）**。假设 <span class="math">$ S = \mathbb{R}^n $</span>，并且我们将状态空间中的每个维度离散化为 <span class="math">$ k $</span> 个值，那么总的离散状态数就是 <span class="math">$ k^n $</span>。随着状态空间维度 <span class="math">$ n $</span> 的增加，离散状态数会以指数级增长，因此这种方法并不适用于大型问题。例如，对于一个 10 维状态空间，如果我们将每个状态变量离散化为 100 个值，那么我们将有 <span class="math">$ 100^{10} = 10^{20} $</span> 个离散状态，这远远超过现代台式机的处理能力。

作为经验法则，离散化通常在一维和二维问题上效果很好（并且有简单且快速实现的优势）。通过一些巧妙的设计和对离散化方法的仔细选择，它在高达四维的状态空间问题上也往往能取得不错的效果。如果你非常聪明并且有一点运气，你甚至可能在六维问题上让它发挥作用。但对于更高维度的问题，这种方法很少奏效。

#### 值函数近似

我们现在描述另一种方法，用于在连续状态的 MDP 中找到策略函数。在这种方法中，我们直接对 <span class="math">$ V^* $</span> 进行近似，而不使用离散化。这种方法被称为 **值函数近似（value function approximation）**，它已经成功地应用于许多强化学习问题。

##### 使用一个模型或模拟器

为了推导值函数近似算法，我们假设我们有一个 MDP 的模型或模拟器。非正式地说，模拟器是一个黑箱，它接受任何（连续值的）状态 <span class="math">$ s_t $</span> 和动作 <span class="math">$ a_t $</span> 作为输入，并根据状态转移概率 <span class="math">$ P_{sa} $</span> 输出下一个状态 <span class="math">$ s_{t+1} $</span> 的采样值。

<div style="text-align: center;">
 <a href="https://s21.ax1x.com/2024/09/22/pAMTSRf.png" data-lightbox="image-13" data-title="simulator">
  <img src="https://s21.ax1x.com/2024/09/22/pAMTSRf.png" alt="simulator" style="width:100%;max-width:375px;cursor:pointer">
 </a>
</div>

有几种方法可以得到这样的模型。一个方法是使用物理仿真。例如，问题集 4 上的倒立摆问题的模拟器是通过使用物理定律来计算在时间 <span class="math">$ t+1 $</span> 时小车/摆杆的位置和姿态，这取决于当前时刻 <span class="math">$ t $</span> 的状态以及采取的动作 <span class="math">$ a $</span>，前提是我们知道系统的所有参数（如摆杆的长度、摆杆的质量等）。或者，也可以使用现成的物理仿真软件包。该软件包接受机械系统的完整物理描述、当前状态 <span class="math">$ s_t $</span> 和动作 <span class="math">$ a_t $</span> 作为输入，并在系统中计算 <span class="math">$ s_{t+1} $</span>，通常以秒为单位的更小时间步长来进行预测。

另一种获得模型的方法是从 MDP 中收集的数据中进行学习。例如，假设我们进行了 <span class="math">$ m $</span> 次试验，在每次试验中，我们在 MDP 中反复执行动作，每次试验运行 <span class="math">$ T $</span> 个时间步长。试验可以通过随机选择动作、执行某个特定策略或其他方式选择动作。我们将观察到如下状态序列：

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

在学习到 <span class="math">$ A $</span> 和 <span class="math">$ B $</span> 之后，我们可以构建一个 **确定性模型（deterministic model）**，其中 <span class="math">$ s_{t+1} = As_t + Ba_t $</span>，并使用该模型来预测未来的状态。

具体来说，我们始终根据公式 (5) 计算 <span class="math">$ s_{t+1} $</span>。或者，我们还可以构建一个 **随机模型（stochastic model）**，在该模型中，<span class="math">$ s_{t+1} $</span> 是关于输入的一个随机函数，其表示形式为：

<div class="math">
$$
s_{t+1} = A s_t + B a_t + \epsilon_t
$$
</div>

其中 <span class="math">$ \epsilon_t $</span> 是噪声项，通常假设 <span class="math">$ \epsilon_t \sim \mathcal{N}(0, \Sigma) $</span>。（协方差矩阵 <span class="math">$ \Sigma $</span> 也可以直接从数据中估计。）

在这里，我们将下一个状态 <span class="math">$ s_{t+1} $</span> 写成当前状态和动作的线性函数；但当然，也可以使用非线性函数。具体来说，可以学习一个模型 <span class="math">$ s_{t+1} = A_{\phi_s} \phi_s(s_t) + B_{\phi_a} \phi_a(a_t) $</span>，其中 <span class="math">$ \phi_s $</span> 和 <span class="math">$ \phi_a $</span> 是状态和动作的某些非线性特征映射。或者，也可以使用非线性学习算法，例如局部加权线性回归 (locally weighted linear regression)，来学习 <span class="math">$ s_{t+1} $</span> 作为 <span class="math">$ s_t $</span> 和 <span class="math">$ a_t $</span> 的函数的估计。

这些方法都可以用于构建确定性或随机模拟器来用于 MDP 的建模。

##### 拟合值迭代

我们现在描述 **拟合值迭代（fitted value iteration）** 算法。该算法用于近似连续状态 MDP 的值函数。在接下来的内容中，我们假设问题的连续状态空间为 <span class="math">$ S = \mathbb{R}^n $</span>，但动作空间 <span class="math">$ A $</span> 是有限且离散的。

回顾值迭代，我们希望执行以下更新：

<div class="math">
$$
\begin{align*}
V(s) &:= R(s) + \gamma \max_{a} \int P_{sa}(s') V(s') ds' \tag{6} \\[5pt]
&= R(s) + \gamma \max_{a} E_{s' \sim P_{sa}}[V(s')] \tag{7}
\end{align*}
$$
</div>

（在第二节中，我们给出了值迭代方程，其中使用的是求和 <span class="math">$ V(s) := R(s) + \gamma \max_a \sum_{s'} P_{sa}(s') V(s') $</span>，而不是积分。新的表示方法意味着我们现在处理的是连续状态而非离散状态。）

拟合值迭代的主要思想是：我们将对有限采样的状态 <span class="math">$ s^{(1)}, \dots, s^{(m)} $</span> 近似执行上述步骤。具体而言，我们将使用监督学习算法——线性回归或如下所述的方法——来近似将值函数表示为状态的线性或非线性函数：

<div class="math">
$$
V(s) = \theta^T \phi(s)
$$
</div>

这里，<span class="math">$ \phi $</span> 是某种适当的状态特征映射。

对于我们的有限状态样本集合中的每个状态 <span class="math">$ s $</span>，拟合值迭代首先计算一个量 <span class="math">$ y(s) $</span>，这是我们对

<div class="math">
$$
R(s) + \gamma \max_a E_{s' \sim P_{sa}}[V(s')]
$$
</div>

的近似（即方程<span class="math">$ (7) $</span>右侧）。然后，它将应用监督学习算法，尝试使 <span class="math">$ V(s) $</span> 接近 

<div class="math">
$$
R(s) + \gamma \max_a E_{s' \sim P_{sa}}
$$
</div>

（或换句话说，尝试使 <span class="math">$ V(s) $</span> 接近 <span class="math">$ y(s) $</span>）

详细地说，算法如下：

> [!TIP]
> <div class="math">
> $$
> \begin{array}{l}
> 1. \text{ Randomly sample } m \text{ states } s^{(1)}, s^{(2)}, \dots, s^{(m)} \in S. \\[5pt]
> 2. \text{ Initialize } \theta := 0. \\[5pt]
> 3. \text{ Repeat } \{ \\[5pt]
> \qquad \text{For } i = 1, \dots, m \{ \\[5pt]
> \qquad \qquad \text{For each action } a \in A \{ \\[5pt]
> \qquad \qquad \qquad \text{Sample } s'_1, \dots, s'_k \sim P_{s^{(i)}a} \text{ (using a model of the MDP).} \\[5pt]
> \qquad \qquad \qquad \text{Set } q(a) = \frac{1}{k} \sum_{j=1}^k R(s^{(i)}) + \gamma V(s'_j). \\[5pt]
> \qquad \qquad \qquad \text{// Hence, } q(a) \text{ is an estimate of } R(s^{(i)}) + \gamma \mathbb{E}_{s' \sim P_{s^{(i)}a}} [V(s')]. \\[5pt]
> \qquad \qquad \} \\[5pt]
> \qquad \qquad \text{Set } y^{(i)} = \max_a q(a). \\[5pt]
> \qquad \qquad \text{// Hence, } y^{(i)} \text{ is an estimate of } R(s^{(i)}) + \gamma \max_a \mathbb{E}_{s' \sim P_{s^{(i)}a}} [V(s')]. \\[5pt]
> \qquad \} \\[5pt]
> \} \\[5pt]
> \text{// In the original value iteration algorithm (over discrete states)} \\[5pt]
> \text{// we updated the value function according to } V(s^{(i)}) := y^{(i)}. \\[5pt]
> \text{// In this algorithm, we want } V(s^{(i)}) \approx y^{(i)}, \text{ which we'll achieve} \\[5pt]
> \text{// using supervised learning (linear regression).} \\[5pt]
> \text{Set } \theta := \arg\min_\theta \displaystyle{\frac{1}{2}} \sum_{i=1}^m \left( \theta^T \phi(s^{(i)}) - y^{(i)} \right)^2. \\[5pt]
> \}
> \end{array}
> $$
> </div>

在上面的算法描述中，我们使用线性回归来实现了拟合值迭代的算法，目的是使 <span class="math">$ V(s^{(i)}) $</span> 接近 <span class="math">$ y(s^{(i)}) $</span>。该算法步骤与标准监督学习（回归）问题完全类似，其中我们有一个训练集 <span class="math">$ (x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(m)}, y^{(m)}) $</span>，并且希望学习一个从 <span class="math">$ x $</span> 到 <span class="math">$ y $</span> 的映射。唯一的区别是，在这里，状态 <span class="math">$ s $</span> 扮演了 <span class="math">$ x $</span> 的角色。尽管我们的描述使用了线性回归，显然其他回归算法（例如局部加权线性回归）也可以使用。

与离散状态上的值迭代不同，拟合值迭代不能被证明总是收敛的。然而，在实际应用过程中，它经常收敛（或近似收敛），并且对于许多问题都能很好地工作。请注意，如果我们使用的是一个确定性模拟器/模型，那么拟合值迭代可以通过设置 <span class="math">$ k = 1 $</span> 来简化。这是因为方程<span class="math">$ (7) $</span>中的期望是关于确定性分布的，因此只需要一个样本就能足够准确地计算该期望。否则，如算法伪代码所示，我们必须采样 <span class="math">$ k $</span> 个样本，并取平均值来近似计算期望（见伪代码中 <span class="math">$ q(a) $</span> 的定义）。

最后，拟合值迭代输出的是 <span class="math">$ V $</span>，这是对 <span class="math">$ V^* $</span> 的一个近似。这一近似值可以明确定义我们的策略函数 <span class="math">$ \pi $</span>。具体来说，当我们的系统处于某个状态 <span class="math">$ s $</span> 时，我们需要选择一个动作 <span class="math">$ a $</span>，并希望能最大化下式：

<div class="math">
$$
\arg \max_a E_{s' \sim P_{sa}}[V(s')] \tag{8}
$$
</div>

在拟合值迭代的内层循环中，我们使用类似的方法来近似这个期望。对于每个动作 <span class="math">$ a $</span>，我们采样 <span class="math">$ s' \sim P_{sa} $</span>，并估计期望。值得注意的是，如果我们的模拟器是下述形式 <span class="math">$ s_{t+1} = f(s_t, a_t) + \epsilon_t $</span>，其中 <span class="math">$ f $</span> 是状态的确定性函数（例如 <span class="math">$ f(s_t, a_t) = A s_t + B a_t $</span>），且 <span class="math">$ \epsilon_t $</span> 是均值为零的高斯噪声，那么在这种情况下，我们可以通过下式选择动作 <span class="math">$ a $</span>：

<div class="math">
$$
\arg \max_a V(f(s, a))
$$
</div>

换句话说，这里我们只是设置了 <span class="math">$ \epsilon_t = 0 $</span>（即忽略模拟器中的噪声），并设置 <span class="math">$ k = 1 $</span>。等价地，我们可以通过使用方程<span class="math">$ (8) $</span>推导出该结果：

<div class="math">
$$
\begin{align*}
E_{s' \sim P_{sa}}[V(s')] &\approx V(E_{s' \sim P_{sa}}[s']) \tag{9} \\[5pt]
&= V(f(s, a)) \tag{10}
\end{align*}
$$
</div>

其中期望是关于随机变量 <span class="math">$ s' \sim P_{sa} $</span> 的。只要噪声项 <span class="math">$ \epsilon_t $</span> 较小，这通常会是一个合理的近似。

然而，对于那些不适用于这种近似的情况，必须对所有 <span class="math">$ |A| $</span> 个动作进行采样，以便使用模型计算上述期望值，这在计算上可能产生非常大的开销。
