# PMIX 统一范式决策与实验重置建议

> 本文档基于 `results_before_Paradigm`、`result_plot/_figure`、历次
> `results_modify_v1` 至 `results_modify_v7` 以及 `实验记录` 中的结果。
> 当前结果主要为 3 seeds 的探索性实验，用于确定范式和搜索空间，不能直接作为论文主结果。

## 1. 结论先行

建议将 PMIX 的主范式固定为：

$$
\begin{aligned}
q_i &= Q_i(\tau_i,u_i),\\
\mathbf z &= E_\psi(c),\\
Q_{\mathrm{tot}}^{(a)}(\mathbf q,c)
&= V_\omega(\mathbf z)
+ M_{\phi_a}^{(a)}(\mathbf q,\mathbf z),\\
a &\in \{\mathrm{MLP},\mathrm{Lattice},\mathrm{KAN}\}.
\end{aligned}
$$

其中只替换部分单调核 $M_{\phi_a}^{(a)}$，并要求

$$
\frac{\partial M_{\phi_a}^{(a)}(\mathbf q,\mathbf z)}{\partial q_i}
\geq 0,
\qquad i=1,\ldots,n.
$$

三个实例统一命名为：

- PMIX-MLP：AMCO 风格的认证部分单调 MLP 核；
- PMIX-Lattice：HLL 风格的认证条件 lattice 核；
- PMIX-KAN：MonoKAN 风格的认证部分单调 KAN 核。

主模型统一使用同一个 $E_\psi$ 和 $V_\omega$，不使用固定 VDN/Q residual，
也不使用任何方法专属的 residual annealing。Q residual 只作为范式级统一消融。

## 2. 为什么选择这个外层公式

### 2.1 $V(\mathbf z)$ 应当成为公共模块

$V(\mathbf z)$ 是与联合动作无关的条件平移项。它不改变动作排序：

$$
\frac{\partial V(\mathbf z)}{\partial q_i}=0.
$$

QMIX 的最后一层 state-dependent bias 本身就承担类似作用。三个 PMIX 都使用相同的
$V(\mathbf z)$，可以把状态价值基线与 action-dependent interaction 分开，避免某个核用大量
容量重复拟合纯状态项。因此，$V$ 应保留，但不能只给某一种方法。

### 2.2 HLL 正尺度应保留在 Lattice 核内部

当前 HLL 的 lattice 输出被限制并中心化，其 action-dependent 动态范围主要依赖
$A(\mathbf z)>0$。直接删除 scale 会人为限制 PMIX-Lattice 的联合价值范围，但 MLP 和 KAN
并没有同样的有界输出限制，因此不需要把该模块提升为 PMIX 的公共组成。

将归一化 lattice 输出记为 $\widetilde M_{\mathrm{Lat}}\in[0,1]$，定义

$$
M_{\mathrm{Lat}}(\mathbf q,\mathbf z)
=A_\xi(\mathbf z)
\left(\widetilde M_{\mathrm{Lat}}(\mathbf q,\mathbf z)-\frac12\right),
\qquad
A_\xi(\mathbf z)>0.
$$

$A_\xi$ 因此是 $M_{\mathrm{Lat}}$ 的架构内 range calibration，而不是外层 PMIX 公式的一部分。
PMIX 统一的是函数约束，不是三个实例完全相同的内部计算图。

### 2.3 单调性与 IGM

固定条件信息 $c$，PMIX 的统一条件为

$$
\frac{\partial Q_{\mathrm{tot}}^{(a)}}{\partial q_i}
=\frac{\partial M_{\phi_a}^{(a)}}{\partial q_i}
\geq 0.
$$

对于 Lattice 特例，进一步有

$$
\frac{\partial M_{\mathrm{Lat}}}{\partial q_i}
=A_\xi(\mathbf z)
\frac{\partial\widetilde M_{\mathrm{Lat}}}{\partial C_i}
\frac{\partial C_i}{\partial q_i}
\geq0,
$$

因为 $A_\xi(\mathbf z)>0$，lattice 插值对坐标 $C_i$ 单调，且 $C_i$ 是非递减校准器。

于是对任意两个仅在第 $i$ 个分量不同的联合动作，若
$Q_i(\tau_i,u_i')\geq Q_i(\tau_i,u_i)$，替换成 $u_i'$ 不会降低
$Q_{\mathrm{tot}}$。逐个替换各 agent 动作可得

$$
\prod_{i=1}^{n}
\underset{u_i}{\arg\max}\ Q_i(\tau_i,u_i)
\subseteq
\underset{\mathbf u}{\arg\max}\ Q_{\mathrm{tot}}(s,\mathbf u),
$$

即保持 IGM 包含关系。若个体最优动作唯一且 $M$ 对每个 $q_i$ 严格递增，则可进一步得到
唯一贪心联合动作的一致性。这里没有要求函数对 $\mathbf z$ 单调，状态可以自由改变 Q 之间的
交互关系。

## 3. 为什么 residual 不进入主范式

候选 residual 为

$$
R_t(\mathbf q)=\lambda(t)\frac{1}{n}\sum_{i=1}^{n}q_i,
\qquad \lambda(t)\geq0.
$$

它确实保持 IGM，因为

$$
\frac{\partial R_t}{\partial q_i}=\frac{\lambda(t)}{n}\geq0.
$$

但它是优化 shortcut，不是条件部分单调逼近所必需的函数结构。历史结果显示：

- `3s_vs_5z` 中，HLL 的 `lattice_size=6`、较高 Q temperature 和 residual 使三个 seed
  更早起飞，说明 residual 能补偿早期弱梯度；
- `MMM2` 中，固定 sum residual 明显过强，容易把 10-agent mixer 推向 VDN-like 局部最优；
- MonoKAN 在 `MMM2` 上的 raw mean、sum 和 tanh residual 均低于 no-residual 版本；
- AMCO 的 residual annealing 比固定 residual 更好，但这是 AMCO 上得到的结果，尚不能证明
  同一 schedule 对 Lattice 和 KAN 都有利。

因此主比较应设置

$$
\lambda(t)\equiv0.
$$

统一 residual 消融可比较：

$$
\lambda(t)=
\lambda_0\max\left(1-\frac{t}{T_{\mathrm{anneal}}},0\right),
$$

建议探索值为 $\lambda_0=0.2$、$T_{\mathrm{anneal}}=10^6$ 环境步，并且必须使用
agent mean 而不是 sum，以消除 agent 数带来的量级变化。该设置必须同时用于三种 PMIX，
可命名为 PMIX-MLP+AR、PMIX-Lattice+AR 和 PMIX-KAN+AR。只有当它在三种核上均表现出稳定的
平均收益时，才适合升级为 PMIX 的公共训练协议。

## 4. 当前结果提供的证据

以下统计由 `results_before_Paradigm` 中每个方法、地图和 seed 的完整 run 计算：
Final 为最后 10 次测试均值，AUC 为约 2M 环境步内的胜率积分。重复 Sacred 目录和明显未完成
run 不重复计入。

| Map | Method | Final | AUC | 主要现象 |
|---|---:|---:|---:|---|
| 1c3s5z | AMCO / HLL / MonoKAN / QMIX | 0.970 / 0.979 / 0.965 / 0.979 | 0.869 / 0.902 / 0.915 / 0.880 | 全部饱和，区分度低 |
| 3s5z | AMCO / HLL / MonoKAN / QMIX | 0.915 / 0.864 / 0.898 / 0.952 | 0.573 / 0.560 / 0.740 / 0.767 | QMIX 与 MonoKAN 学得更快 |
| 5m_vs_6m | AMCO / HLL / MonoKAN / QMIX | 0.599 / 0.691 / 0.748 / 0.502 | 0.415 / 0.430 / 0.501 / 0.365 | MonoKAN 最稳，QMIX 方差大 |
| 2c_vs_64zg | AMCO / HLL / MonoKAN / QMIX | 0.919 / 0.904 / 0.889 / 0.946 | 0.466 / 0.609 / 0.652 / 0.676 | AMCO 明显晚起飞，QMIX 最稳 |
| 27m_vs_30m | AMCO / HLL / MonoKAN / QMIX | 0.529 / 0.191 / 0.430 / 0.515 | 0.234 / 0.052 / 0.190 / 0.199 | AMCO 有最高 seed 也有失败 seed；HLL 分组扩展明显受限 |
| bane_vs_bane | AMCO / HLL / MonoKAN / QMIX | 1.000 / 1.000 / 0.998 / 0.564 | 0.858 / 0.945 / 0.837 / 0.513 | PMIX 均强，但 QMIX seed 方差极大 |

历史 best 曲线还显示：

- `3s_vs_5z` 上三种 PMIX 在约 0.5M 至 0.8M 开始学习，QMIX 大约晚 0.5M 至 0.8M；
- `MMM2` 上 AMCO annealed-residual 的 best selection 高于其他方法，但它来自不同调参历史，
  不能作为公平主结果；
- `3s5z` 和 `2c_vs_64zg` 中 QMIX 并未被全面超过，所以论文不应声称 PMIX 在所有任务都优于
  hypernetwork；更稳妥且更准确的主张是 PMIX 提供更直接的条件交互函数类，并在部分地图上
  改善样本效率或稳定性。

这些结果支持“统一重置范式”，而不支持把各地图历史最优增强拼接成最终模型。

## 5. 哪些差异允许保留在各架构内部

公平不等于三种核使用完全相同的内部参数。以下差异是逼近器定义或计算复杂度的一部分，
可以保留：

| 组件 | PMIX-MLP | PMIX-Lattice | PMIX-KAN |
|---|---|---|---|
| 单调认证 | Q 分支非负权重，state 分支自由 | 有序顶点与单调插值 | 单调 spline 边及认证组合 |
| 核容量 | depth、width | lattice resolution、顶点预算 | hidden width、grid size |
| Q 坐标处理 | 核内连续单调映射或 raw Q | sigmoid 后映射到 lattice 坐标 | tanh 后进入固定 spline grid |
| 输出范围 | 由单调 MLP 直接学习 | 正条件尺度对 $[0,1]$ 输出反归一化 | 由 KAN 核直接学习 |
| 大规模处理 | 普通全连接 | 确定性 grouping | 普通 KAN 组合 |

以下内容不能再作为某个模型的专属增强：

- 独立 $V(s)$；
- VDN/Q residual 或 residual annealing；
- 只对 AMCO 使用的 map-specific state branch scale；
- 只对某个模型增加的额外 state encoder 深度；
- 根据测试结果人工指定的逐地图超参数表。

### 5.1 HLL 的确定性容量规则

HLL 的顶点数随单调输入维度指数增长，这是架构内在性质，不是可选增强。建议固定顶点预算
$B=1024$，并仅按单调输入维数设置 resolution 上限

$$
\kappa(g)=
\begin{cases}
16,&g\le2,\\
6,&g\ge3.
\end{cases}
$$

再令

$$
g=\min(n,10),
\qquad
k=\max\left(2,\min\left(\kappa(g),
\left\lfloor B^{1/g}\right\rfloor\right)\right).
$$

该规则给出：2 agents 使用 $16^2=256$ 个顶点，3 agents 使用 $6^3=216$ 个顶点，
5 agents 使用 $4^5=1024$ 个顶点，
8 至 10 个输入组使用 binary lattice。超过 10 agents 时固定分成 10 组：同质队伍使用
sorted-quantile mean；异质队伍按 unit type 后再做固定连续分组。规则只依赖 agent 数和类型，
不能依赖某张地图跑出的胜率。

需要明确报告：grouping 后的 PMIX-Lattice 仍单调，但它不再具有完整 $n$ 维 lattice 的表达能力。
`27m_vs_30m` 中 HLL 的低 AUC 正是这一扩展性代价的实证信号，不应通过隐去或专属 shortcut
来掩盖。

### 5.2 “Beyond Hypernetworks” 的表述边界

当前 HLL 辅助网络根据状态一次输出全部 lattice 顶点值。从计算图外观来看，审稿人可能将其
理解为另一种 conditional coefficient generator。理论上应将其严格写成固定参数函数

$$
g_\theta(\mathbf z,v),
$$

在每个固定顶点 $v$ 上的并行求值；顶点值是 $g_\theta$ 的激活输出，而不是状态生成的可学习
参数。即使如此，论文也不应作“PMIX 完全不包含任何条件系数计算”的过强声明。更准确的
主张是：PMIX 将问题提升为条件部分单调函数逼近，并超越 QMIX-style 的 state-to-mixer-weight
参数化。Hyper-MLP 与 Direct-MLP 控制实验负责单独验证 direct conditioning 的作用。

## 6. 建议的统一初始实现

### 6.1 公共模块

建议三个方法先统一为：

| 项目 | 建议值 |
|---|---|
| context encoder $E_\psi$ | 2-layer MLP，hidden/output 64，SiLU |
| state value $V_\omega$ | 64 -> 32 -> 1，ReLU |
| Q residual | 0 |
| agent network / learner | 与 QMIX 完全相同 |
| replay / target / epsilon | 与 QMIX 完全相同 |
| map-specific overrides | 全部删除，HLL 确定性容量规则除外 |

所有核只接收同一个 $\mathbf z=E_\psi(c)$，$V$ 也从该 $\mathbf z$ 计算。不要让 HLL
直接使用 raw state、AMCO 使用 104 维 encoder、MonoKAN 使用另一个 32/64 维 encoder，否则状态
表示能力仍然混在架构比较中。

PMIX-Lattice 在其核内部额外使用
$A_\xi(\mathbf z)=10^{-3}+\operatorname{softplus}(a_\xi(\mathbf z))$，建议将其初始化为接近 1。
该模块必须计入 Lattice 的参数量和架构说明，但不复制到 MLP 或 KAN。

### 6.2 核的初始容量

可将以下值作为统一重跑前的起点，而不是最终论文超参数：

| 核 | 初始设置 | 原因 |
|---|---|---|
| PMIX-MLP | depth 4，width 64，单调激活 ReLU/ELU | 保留 AMCO 认证结构，去掉专属 state scale |
| PMIX-Lattice | 上述 $B=1024,\kappa(g)$ 规则；Q temperature 先全局固定 | 复现小 n 需要高 resolution、大 n 需要 binary/grouping 的规律 |
| PMIX-KAN | hidden 32，grid 7，noise 0.02，temperature 1 | 历史上小容量 no-residual 最稳定，扩大 grid/hidden 增加 seed 方差 |

HLL 的全局 temperature 建议只在开发地图上比较 $\{1,2\}$；MLP/KAN 也应获得同等数量的
一个核心超参数候选。选取依据必须是开发地图跨 seed 的 aggregate AUC，而不是分别为每张地图
选择最好值。

## 7. 统一实验顺序

### 阶段 A：范式确认，不进入论文主表

在 `3s_vs_5z` 和 `MMM2` 上用相同 3 seeds 比较：

1. 公共 $E+V$、无 residual；
2. 公共 $E+V$、统一 annealed mean residual。

选择规则应预先固定为两张开发地图 normalized AUC 的平均 rank，并同时检查失败 seed 数。
不能让 MLP 选 no-residual、Lattice 选 residual、KAN 再选另一种版本。

另在 PMIX-Lattice 内部比较 state-dependent、global-positive 和 no-scale 三种 range
calibration。该实验是 Lattice 架构消融，不参与三个 PMIX 的公共范式选择。

### 阶段 B：冻结范式后的正式主实验

范式确定后冻结所有公共模块和 schedule，在新 seeds 上运行：

- 核心地图：`3s_vs_5z`、`5m_vs_6m`、`3s5z`、`MMM2`、`2c_vs_64zg`、
  `27m_vs_30m`；
- sanity/附录：`2s3z`、`1c3s5z`、`bane_vs_bane`；
- 至少 5 seeds，理想为 8 seeds；
- 同时报告 median/mean、95% bootstrap CI、AUC、最终窗口、time-to-threshold 和失败率。

`1c3s5z`、`2s3z` 等饱和地图不应主导平均分；`27m_vs_30m` 必须保留，因为它揭示了
Lattice 的扩展性边界和 AMCO 的 seed instability。

### 阶段 C：支撑 “Beyond Hypernetworks” 的关键控制

最终还需要一个参数量匹配的 Hyper-MLP 与 Direct-MLP 对照。两者共享相同的 $E$、$V$、
深度、宽度和参数预算，唯一差异是状态用于生成单调核参数，还是作为自由输入直接参与函数
求值。否则主实验只能证明三种 PMIX 实例是有效替代方案，不能把全部提升严格归因于
“direct conditioning 优于 hypernetwork conditioning”。

## 8. 最终边界

论文主方法应当是一个干净、可证明、可替换的函数范式：

$$
\boxed{
Q_{\mathrm{tot}}^{(a)}
=V(\mathbf z)+M^{(a)}(\mathbf q,\mathbf z),
\quad \partial M^{(a)}/\partial q_i\geq0
}
$$

其中：

- $E$ 和 $V$ 是 PMIX 公共协议；
- MLP、Lattice、KAN 只定义不同的认证部分单调核；
- $A_\xi(\mathbf z)>0$ 只属于有界 Lattice 核的 range calibration；
- residual、annealing 和其他 shortcut 是统一训练消融，不属于某个变式；
- HLL resolution/grouping 是由固定计算预算决定的架构规则，不是逐地图调参；
- 正式结果必须来自冻结配置和新 seeds，历史 best-of-tuning 曲线只用于形成假设。
