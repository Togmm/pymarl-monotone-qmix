# PMIX 实验设计方案

> 对应理论文档：[PMIX_theory_framework.md](PMIX_theory_framework.md)
>
> 本文档用于规划正式论文实验。当前 results_origin、results_modify_v1 至 results_modify_v7 中的结果应视为探索性实验，用来形成假设和确定合理搜索范围，不直接作为最终无偏主结果。

## 1. 实验需要回答的核心问题

实验不应围绕“哪个模型经过充分调参后曲线最高”组织，而应围绕以下 Research Questions（RQ）组织。

### RQ1：PMIX 是否优于 hypernetwork-based mixing？

需要回答：

> 在保持 IGM、agent network、训练算法和模型容量基本一致时，将状态作为自由输入直接参与部分单调函数，是否比使用状态生成 mixer 参数具有更好的性能、样本效率或稳定性？

这是标题 “Beyond Hypernetworks” 最关键的实验问题。

### RQ2：不同 PMIX 逼近器具有怎样的归纳偏置？

比较：

- PMIX-MLP；
- PMIX-Lattice；
- PMIX-KAN。

重点不是证明其中一个在所有地图都最好，而是解释：

- MLP 是否在高维和异质场景更稳定；
- Lattice 是否在低维、阈值明显的场景更有效；
- KAN 是否更适合局部非线性和状态依赖阈值；
- 各方法随 agent 数增加时如何退化。

### RQ3：提升是否真正来自状态-Q交互？

需要排除以下替代解释：

- 参数量更大；
- 多了独立的 $V(s)$；
- HLL 多了架构内部 positive output scale；
- 多了 VDN/Q residual；
- 使用了更多逐地图超参数；
- 某些模型只是在早期获得更强梯度。

### RQ4：PMIX 是否在实践中始终保持单调性和 IGM？

理论保证需要配合数值验证：

$$
\min_i\frac{\partial Q_{\mathrm{tot}}}{\partial Q_i}\ge0.
$$

不仅检查最终 checkpoint，还应在训练期间定期检查。

### RQ5：PMIX 的代价是什么？

报告：

- 参数量；
- 单次训练更新时间；
- 环境步吞吐；
- GPU 峰值显存；
- 随 agent 数增加的时间和空间变化；
- HLL 的 lattice/grouping 复杂度。

---

## 2. 方法与基线

### 2.1 主比较方法

主表至少包含：

1. **VDN**：加性分解下界和 residual shortcut 参照。
2. **QMIX**：核心 hypernetwork baseline。
3. **PMIX-MLP**：当前 AMCO 路线重命名并统一公共模块。
4. **PMIX-Lattice**：当前 HLL 路线。
5. **PMIX-KAN**：当前 MonoKAN 路线。

如果实现成本允许，可增加一个现代 IGM baseline，例如 QPLEX。但新增 baseline 不能挤占三种 PMIX 和 QMIX 的 seed 预算。当前论文的最低完整配置仍是上述五种方法。

### 2.2 必须增加的 conditioning-control baseline

仅比较 QMIX 和 PMIX-MLP 仍不能完全隔离“hypernetwork vs direct context”，因为两者的 mixer 结构也不同。

建议实现一组参数匹配的控制实验：

- **Hyper-MLP**：状态生成单调 MLP 的权重或调制参数；
- **Direct-MLP / PMIX-MLP**：同等深度、宽度和参数预算，状态作为普通自由输入。

二者共享：

- 相同 agent network；
- 相同状态编码维度；
- 相同层数和隐藏宽度；
- 相同 $V(s)$；
- 相同公共外层结构；
- 相同参数预算与优化器。

该对照是支撑标题的最重要实验。若暂时无法实现，论文结论应写成“PMIX provides an effective alternative to hypernetwork mixing”，而不要写成“direct context is intrinsically superior”。

---

## 3. 公平性协议

### 3.1 统一外层结构

三个 PMIX 实例统一写为

$$
Q_{\mathrm{tot}}^{(a)}
=V_\omega(E_\psi(s))
+M_{\phi_a}^{(a)}(\mathbf q,E_\psi(s)),
\qquad
a\in\{\mathrm{MLP},\mathrm{Lat},\mathrm{KAN}\}.
$$

主实验中必须统一：

- 状态编码器 $E_\psi$ 的层数、激活和输出维度；
- $V_\omega$ 的层数和宽度；
- 是否使用 Q residual；
- agent network、learner 和 TD loss；
- replay buffer、target update、epsilon schedule；
- 训练与评估预算。

### 3.2 关于 $V(s)$

当前代码中不同 mixer 对 state_value 的实际使用并不一致。正式实验前应保证

$$
V_\omega(s)
$$

要么在三个 PMIX 实例中全部启用，要么全部关闭并作为统一消融。

推荐主模型全部启用，因为 QMIX 本身具有状态价值基线。另做一次 with $V(s)$ / without $V(s)$ 消融即可。

### 3.3 关于 state-dependent output scale

当前 HLL 独有

$$
A_\xi(s)>0.
$$

它会直接改变 $Q_{\mathrm{tot}}$ 对个体 Q 的敏感度，但其直接作用是把 HLL 的有界、
中心化输出恢复到联合价值所需的动态范围。因此将它定义为 PMIX-Lattice 核内部的
range calibration：

$$
M_{\mathrm{Lat}}(\mathbf q,z)
=A_\xi(z)\left(\widetilde M_{\mathrm{Lat}}(\mathbf q,z)-\frac12\right),
\qquad A_\xi(z)>0.
$$

$A_\xi$ 不加入 PMIX-MLP 和 PMIX-KAN，也不出现在 PMIX 的统一外层公式中。实验应报告
该模块及其初始化，并在 HLL 内部做 state-dependent scale、global positive scale 或合理的
无 scale 对照，确认收益来自解除输出范围限制。不能把它与 VDN/Q residual 混为同类：
前者是 HLL 原生表示的校准，后者是绕过主核的额外信用通路。完整依据见
[PMIX_paradigm_decision.md](PMIX_paradigm_decision.md)。

### 3.4 关于 Q residual

主比较建议不使用 Q residual：

$$
R(\mathbf q)=0.
$$

原因是现有实验已经表明 residual 会显著改变优化路径，而且对 AMCO、HLL、MonoKAN 的影响不同。

将 residual 作为统一机制消融：

$$
Q_{\mathrm{tot}}^{(+R)}
=Q_{\mathrm{tot}}
+\lambda(t)\frac{1}{n}\sum_iQ_i.
$$

比较：

- no residual；
- fixed residual；
- annealed residual。

三种 PMIX 必须使用相同的初值、终值和退火时间。现有 AMCO residual annealing 结果可以作为提出该消融的依据，但不能作为 PMIX-MLP 主模型的专属增强。

### 3.5 参数量和计算量

建议将总 mixer 参数量控制在 QMIX 的 $\pm5\%$ 内。若某架构无法同时匹配参数量和计算量，至少报告：

- mixer 参数量；
- 总参数量；
- 每个 learner update 的平均耗时；
- 每百万环境步 wall-clock 时间；
- GPU 峰值显存。

HLL 顶点数随 agent 数指数增长，因此参数匹配不能只看辅助网络参数，还必须报告实际顶点数和插值计算量。

### 3.6 禁止不对称的逐地图人工调参

当前 YAML 中包含大量 map-specific 配置，适合作为探索阶段，但容易造成方法间调参预算不对称。

正式主实验推荐：

1. 在开发地图上确定一套全局默认配置；
2. 冻结配置；
3. 在主评估地图和新 seeds 上运行；
4. 仅允许与规模有关的确定性规则，例如 HLL 根据最大顶点预算自动分组。

如果必须逐地图调参，每种方法必须使用相同搜索次数、相同候选规模和相同选择指标。

---

## 4. 地图选择

不建议把大量接近 100% 胜率的简单地图作为主结论来源。地图应由研究假设驱动。

### 4.1 推荐的主地图集合

| 地图 | 主要属性 | 用途 |
|---|---|---|
| 3s_vs_5z | 3 agents、走位和局部阈值明显 | 检验低维非线性和早期学习 |
| 5m_vs_6m | 中等难度、seed 方差明显 | 检验稳定性和样本效率 |
| 3s5z | 多类型协作、中等 agent 数 | 检验异质协作 |
| MMM2 | 10 agents、强异质性、hard | 核心复杂协作地图 |
| 2c_vs_64zg | 极端单位数量差异和集火需求 | 检验非加性联合策略 |
| 27m_vs_30m | 大规模同质 agents | 检验扩展性 |

如果计算预算不足，最低主集合建议为：

- 3s_vs_5z；
- 5m_vs_6m；
- MMM2；
- 27m_vs_30m。

### 4.2 辅助或附录地图

- 2s3z、1c3s5z：容易饱和，适合 sanity check；
- bane_vs_bane：适合大规模/分组策略分析；
- 其他已有地图可放附录完整结果，不必全部进入主表。

### 4.3 地图分组报告

除逐地图曲线外，可按属性汇总：

- small-agent；
- heterogeneous coordination；
- large-agent scalability；
- hard exploration。

这样能够解释三种 PMIX 的归纳偏置，而不是只报告平均胜率。

### 4.4 如何理解地图特征与方法适配

这里的“适配”不是指某种方法在理论上不能表示其他地图，而是指其函数参数化是否更容易从有限的 TD 样本中学出有效的 credit assignment。所有 PMIX 变式都满足对个体价值的单调约束，但不同函数族对

$$
\frac{\partial Q_{\mathrm{tot}}}{\partial Q_i}
$$

施加了不同的结构偏置。分析地图时应重点考虑以下六个维度。

#### 4.4.1 己方 agent 数量与 mixer 输入维度

Mixer 的单调输入是己方个体价值

$$
\mathbf Q=(Q_1,\ldots,Q_n),
$$

因此真正决定 mixer 输入维度的是己方 agent 数，而不是敌方单位总数。例如，`3s_vs_5z` 的 mixer 是三维，`2c_vs_64zg` 虽然包含 64 个敌人，但己方只有两个 Colossus，因此 mixer 仍然是二维；`MMM2` 有 10 个己方单位，因此是十维。

这一点对 HLL 尤其重要。若每个维度使用 $K$ 个格点，完整 lattice 的顶点数为

$$
N_{\mathrm{vertices}}=K^n.
$$

随着 agent 数增加，完整 lattice 的计算和参数规模呈指数增长。大规模任务必须降低 lattice resolution 或进行 grouping，而 grouping 又可能损失个体身份和细粒度 credit 信息。

#### 4.4.2 同质性与角色异质性

同质地图中的己方单位类型相同，例如 `5m_vs_6m` 中的五个 Marine。它们具有相同的射程、速度和基本职责，credit 的差别主要来自当前血量、位置、目标和承伤情况。

异质地图则包含不同单位类型。例如 `MMM2` 中 Marine、Marauder 和 Medivac 分别承担输出、承伤和治疗职责。此时 mixer 不仅需要判断某个动作是否有价值，还需要识别该 agent 的角色以及该角色在当前战斗阶段的重要性。因此，能够保留有序 agent identity 并按状态改变个体权重的 QMIX，通常在强异质地图上具有较自然的归纳偏置。

#### 4.4.3 对称性与非对称性

`2s3z`、`3s5z` 和 `1c3s5z` 的双方兵力构成基本相同，胜负主要由集火、阵型和伤害交换效率决定。`3s_vs_5z`、`5m_vs_6m`、`2c_vs_64zg` 和 `MMM2` 则存在数量或类型劣势，通常不能依靠正面对攻取胜，而需要利用射程、移动、治疗或范围伤害等机制。

非对称地图的有效策略区域通常更窄。模型不仅要学会造成伤害，还必须发现一套能够持续获胜的联合策略，因此更容易出现长时间零胜率后突然上升的学习曲线。

#### 4.4.4 局部阈值与策略切换

部分战斗决策具有明显阈值，例如进入攻击射程、即将被近战单位追上、血量降低到必须撤退或敌方血量降低到可以一次集火击杀。在阈值附近，很小的状态变化可能使最优动作从攻击切换为撤退。

这类任务与 PMIX-KAN 的样条表示较匹配。样条可以在普通输入区间保持平缓，在关键区间快速变化，使

$$
\frac{\partial Q_{\mathrm{tot}}}{\partial Q_i}
$$

在关键决策区域增大，而不要求整个函数保持同一斜率。

#### 4.4.5 局部 credit 与全局角色依赖

局部 credit 主要由 agent 自己的射程、血量、目标和受包围程度决定。PMIX-KAN 通常更容易表示这种局部阈值型关系，HLL 则适合表示规则的分段联合关系。

全局 credit 表示某个 agent 的价值高度依赖整个团队状态。例如 Medivac 的治疗动作是否有价值，取决于哪些队友受伤、这些队友是否处在关键位置以及治疗后能否继续输出。QMIX 可以通过状态条件权重直接改变不同角色的重要性；PMIX-MLP 也能够表示复杂的全局交互，但需要从 TD 信号中自行发现这种分解，因此通常更难优化。

#### 4.4.6 反馈密度与有效策略可发现性

频繁的伤害和击杀事件能够提供较多中间反馈，但中间反馈密集并不意味着获胜策略容易发现。例如 `3s_vs_5z` 中，Stalker 即使造成一定伤害，如果没有形成可持续的攻击--撤退循环，最终仍会失败。

因此应分别考察模型能否降低 TD loss、能否提高伤害，以及能否形成稳定胜率。长时间接近零胜率后突然上升，通常意味着模型很晚才建立出能够支持完整获胜策略的 credit 通路。

### 4.5 当前地图的具体特征

#### 4.5.1 `1c3s5z`

双方通常各包含 1 个 Colossus、3 个 Stalker 和 5 个 Zealot，是一个九 agent、对称且异质的地图。

- Colossus 是远程范围输出核心；
- Stalker 提供远程持续输出；
- Zealot 是近战前排，负责接敌和承担伤害。

主要协作关系是 Zealot 形成前排，使 Stalker 和 Colossus 保持安全输出。该地图存在角色差异，但合理策略较多，最终容易接近饱和。因此它更适合比较早期样本效率和稳定性，不适合单独支撑表达能力优势。MonoKAN 可以快速学习射程和血量阈值，QMIX 可以按状态突出 Colossus 等关键角色，而 HLL 和 PMIX-MLP 最终也能充分学习。

#### 4.5.2 `2s3z`

双方通常各包含 2 个 Stalker 和 3 个 Zealot，是一个五 agent、对称异质地图。主要要求包括 Zealot 前排接战、Stalker 后排输出以及全队集火同一目标。

该地图的联合价值结构较规则：任意单位动作价值提高通常都会提高团队价值，而前排和后排同时发挥作用时会产生协同收益。它不要求长时间精确 kiting，因此 HLL 的分段多线性联合价值面和 QMIX 的状态条件加权都较容易建立有效 credit。最终各方法接近饱和，说明 mixer 表达能力不是主要瓶颈。

#### 4.5.3 `3s5z`

双方通常各包含 3 个 Stalker 和 5 个 Zealot，是一个八 agent、对称异质地图。相比 `2s3z`，它具有更大的联合动作空间、更多集火组合以及更复杂的前后排站位。

即使单位类型相同，不同 agent 也会因血量、距离和位置而具有不同的重要性。因此，“当前状态下具体哪个 agent 更关键”成为主要 credit 问题。QMIX 能够为每个有序 agent 生成独立的状态条件权重，具有明显优势；MonoKAN 可以处理射程、血量和攻击阈值；HLL 受到高维 lattice resolution 限制；PMIX-MLP 虽然表达能力充足，但需要在更大的联合空间中自行发现 credit 结构，因而通常启动较慢、seed 方差更大。

#### 4.5.4 `3s_vs_5z`

己方为 3 个 Stalker，敌方为 5 个 Zealot，是一个低维、非对称的经典 kiting 地图。Stalker 是远程单位，Zealot 是近战单位；如果 Stalker 停在原地攻击，数量更多的 Zealot 接近后会迅速造成大量伤害。

有效策略需要重复执行攻击、后退拉开距离、再次攻击的循环，并保持三个 Stalker 基本同步、集中攻击同一目标和避免单个单位被包围。最优动作高度依赖攻击射程、危险距离、自身血量和敌方残血等局部阈值。

因此该地图最符合 MonoKAN 的局部样条归纳偏置。PMIX-MLP 最终也能表示复杂的状态--Q 交互，但需要更多样本。QMIX 可以改变 agent 权重，却不一定能最直接地表示距离阈值附近的精细价值曲线。HLL 的问题不应简单归因于格点数量不足，更可能与关键区域的 lattice slope、sigmoid 校准饱和、output scale 和状态价值分支主导有关。

#### 4.5.5 `5m_vs_6m`

己方为 5 个 Marine，敌方为 6 个 Marine，是一个同质、非对称、以少打多的地图。Marine 是远程、低血量输出单位，双方没有固定的坦克、治疗或后排角色。

主要协作要求是集火、快速减少敌方数量、让残血单位及时后退并避免多个单位分别攻击不同目标。Agent 之间的 credit 差异主要来自瞬时血量、位置、目标和承伤情况，而不是固定角色。

这类平滑但具有局部阈值的同质协作与 MonoKAN 较匹配；HLL 也可以直接表示多个 Marine 同时采取高价值动作时的联合收益。QMIX 的固定有序 agent identity 优势较弱，因为哪个 Marine 更重要会随位置和血量不断变化。PMIX-MLP 的自由度可能超过任务实际需要，从而增加优化成本。

#### 4.5.6 `2c_vs_64zg`

己方为 2 个 Colossus，敌方为 64 个 Zergling。Colossus 具有远程范围伤害，Zergling 则是数量极多、血量较低的近战单位。两个 Colossus 必须利用范围攻击、保持距离并避免被敌群包围。

尽管环境中敌人很多，mixer 仍然只有两个 Q 输入：

$$
\mathbf Q=(Q_1,Q_2).
$$

64 个敌人的分布主要进入全局状态 $z$。因此该地图的特点不是 mixer 维度高，而是状态复杂且状态条件 credit 非常重要。QMIX 可以根据敌群分布直接改变两个 Colossus 的权重，因而通常学习较快；MonoKAN 可以处理距离、包围程度和攻击时机等阈值；二维 HLL 的理论容量充足，但需要从复杂状态中生成合适的顶点值，早期优化可能较慢。频繁击杀 Zergling 提供了较密集的反馈，使较慢的方法后期仍能达到较高胜率。

#### 4.5.7 `MMM2`

己方通常包含 7 个 Marine、2 个 Marauder 和 1 个 Medivac，共 10 个 agent；敌方通常包含 8 个 Marine、3 个 Marauder 和 1 个 Medivac，共 12 个单位。这是一个高度异质、非对称且具有长时序协作要求的困难地图。

- Marine 负责主要输出，但血量较低；
- Marauder 血量更高，可承担更多伤害；
- Medivac 负责治疗，其价值取决于其他单位的血量、位置和存活状态。

任务同时要求集火、保护 Medivac、选择治疗目标、让 Marauder 合理承伤、残血单位后退以及优先处理敌方关键单位。这里的 credit 是典型的全局、高阶、角色依赖关系。例如

$$
\frac{\partial Q_{\mathrm{tot}}}{\partial Q_{\mathrm{Medivac}}}
$$

会随所有输出单位的血量和位置变化。QMIX 具有状态条件角色加权优势；PMIX-MLP 理论上能够表示复杂高阶交互，但优化困难；完整二值 HLL 具有 $2^{10}=1024$ 个顶点，可以直接编码较强的高阶联合关系；MonoKAN 适合治疗触发、残血撤退和射程控制等局部阈值，但全局角色协作并不是其最自然的优势。

当前 `paradigm_map_mmm2_test_win.png` 中仅 PMIX-MLP/AMCO 是新的 residual-free 运行，HLL、MonoKAN 和 QMIX 曲线来自旧实验，因此该图只能用于提出机制假设，不能作为冻结协议下的严格性能结论。

### 4.6 地图特征与方法适配汇总

| 地图 | 核心特征 | 主要 credit 类型 | 相对匹配的方法 |
|---|---|---|---|
| `1c3s5z` | 对称、异质、容易饱和 | 角色分工与基本前后排协作 | MonoKAN、QMIX；各方法最终均可学好 |
| `2s3z` | 小规模、对称、规则协作 | 分段规则、接近加性的联合 credit | HLL、QMIX |
| `3s5z` | 中等规模、异质、集火组合增多 | 状态条件的个体重要性 | QMIX、MonoKAN |
| `3s_vs_5z` | 小规模、非对称、精确 kiting | 距离、血量和动作切换阈值 | MonoKAN、PMIX-MLP |
| `5m_vs_6m` | 同质、以少打多 | 血量、距离和集火的局部关系 | MonoKAN、HLL |
| `2c_vs_64zg` | 低 Q 维度、复杂敌方状态、范围伤害 | 强状态条件 credit | QMIX、MonoKAN |
| `MMM2` | 高度异质、十 agent、长时序协作 | 全局角色依赖和高阶交互 | QMIX、HLL 具有结构潜力，需统一重跑确认 |

从归纳偏置角度，可以将四类 mixer 概括为：

- PMIX-KAN 更自然地回答“什么时候应该突然改变策略”；
- PMIX-Lattice 更自然地回答“不同联合价值区域中，agent 如何产生协同收益”；
- PMIX-MLP 更自然地回答“状态和所有个体价值之间是否存在复杂且无固定形式的交互”；
- QMIX 更自然地回答“当前状态下，哪个 agent 或角色更加重要”。

这些是基于架构和当前学习曲线得到的机制解释，而不是严格的能力边界或因果证明。正式论文应通过 $\partial Q_{\mathrm{tot}}/\partial Q_i$ 分布、状态价值与 mixing 分支幅值比、HLL lattice slope、MonoKAN 校准饱和率等内部诊断进一步验证。

---

## 5. Seeds、预算与评估

### 5.1 Seeds

三 seeds 只能用于开发和快速筛选。正式主结果建议：

- 最低：5 seeds；
- 推荐：8 seeds；
- 计算充足：10 seeds。

所有方法、所有地图使用完全相同的预先固定 seeds。开发阶段 seeds 与最终报告 seeds 应分离。

一种可行设置：

- 调参 seeds：1、41、141；
- 最终评估：另选 5-8 个固定 seeds，正式运行前写入协议。

不要在看到结果后删除失败 seed 或替换 seed。

### 5.2 训练预算

当前约 2.05M 环境步对部分 hard/large 地图可能不足。

建议：

- 中等地图：2M steps；
- hard/large 地图：5M steps；
- 同一地图上所有方法使用相同预算。

如果所有地图统一 2M，应将结论解释为 fixed-budget sample efficiency，而不能断言最终收敛性能。

### 5.3 测试频率和测试 episode

建议：

- 每 10K 或 20K 环境步测试一次；
- 每次至少 32 个 test episodes；
- 资源允许时使用 64 个；
- 所有方法保持一致。

### 5.4 最终性能定义

不要使用单条曲线的最后一个点或 peak。

推荐每个 run 的最终性能定义为最后 $K$ 次评估的平均：

$$
\operatorname{Final}(r)
=
\frac{1}{K}\sum_{j=T-K+1}^{T}w_{r,j},
\qquad K=5\text{ 或 }10.
$$

Peak 只能作为补充指标，因为它对评估噪声和 checkpoint 选择高度敏感。

### 5.5 样本效率

报告归一化 AUC：

$$
\operatorname{AUC}(r)
=
\frac{1}{T}
\int_0^T w_r(t)\,dt.
$$

同时报告达到固定胜率阈值的环境步数，例如

$$
T_{50},\quad T_{75},\quad T_{90}.
$$

对从未达到阈值的 run 使用 censored/not reached 标记，不应直接填成训练终点后再做普通均值。

---

## 6. 统计报告方式

### 6.1 单地图结果

每张地图报告：

- seed 中位数曲线；
- 25%-75% 区间或 bootstrap 95% CI；
- final performance；
- normalized AUC；
- 成功 seed 比例。

不要只画挑选出的 best seed 或 best variant。

### 6.2 跨地图汇总

推荐使用：

- Interquartile Mean（IQM）；
- stratified bootstrap 95% confidence interval；
- performance profile；
- probability of improvement over QMIX。

跨地图前需要对分数归一化，避免简单地图的 0.01 差异和困难地图的 0.1 差异被错误等权解释。

### 6.3 方法选择偏差

如果同一方法测试了多个版本，再从中选择最好版本，其有效调参预算已经高于只运行一个版本的 baseline。

因此正式结果必须：

1. 在调参数据上选择配置；
2. 冻结配置；
3. 在未参与选择的 seeds 上报告；
4. 披露每种方法尝试过的配置数。

---

## 7. 核心消融实验

消融不需要覆盖所有地图。推荐在两个代表性地图上进行：

- 3s_vs_5z：低维、阈值型；
- MMM2：高维、异质型。

每个消融至少 5 seeds。

### 7.1 Conditioning mechanism

最重要的消融：

| 模型 | 状态使用方式 | 单调约束 |
|---|---|---|
| QMIX | 状态生成 mixer 参数 | 对 Q 单调 |
| Hyper-MLP | 状态生成匹配 MLP 参数/调制 | 对 Q 单调 |
| PMIX-MLP | 状态作为直接自由输入 | 对 Q 单调 |
| PMIX-MLP w/o context interaction | 仅 $M(\mathbf q)+V(s)$ | 对 Q 单调 |

最后一项用于区分 $V(s)$ 带来的状态基线与 $M(\mathbf q,z)$ 带来的真实状态-Q交互。

### 7.2 Common components

统一比较：

- with/without $V(s)$；
- no/fixed/annealed Q residual；
- raw state vs shared state encoder。

这些消融必须对三个 PMIX 使用同一设置。

### 7.3 Capacity

为每种 PMIX 设置 small/base/large 三档，并尽量参数匹配：

- MLP：hidden width/depth；
- Lattice：group size、lattice resolution；
- KAN：hidden width、grid size。

观察性能是否来自更大容量，以及增加容量是否放大 seed 方差。

### 7.4 架构内部消融

这部分可放附录，不必强求完全一致：

- PMIX-Lattice：full vs grouped、不同 grouping；
- PMIX-Lattice：state-dependent/global/no output scale；
- PMIX-KAN：grid、temperature、noise、输入校准；
- PMIX-MLP：state input scale、activation-switch vs 纯非负权重。

内部消融用于解释各架构，不应进入三种 PMIX 的公平主比较。

---

## 8. 机制诊断

仅报告胜率不足以证明 PMIX 改善了状态-Q交互。

### 8.1 单调性数值验证

定期从 replay buffer 采样 $(\mathbf q,s)$，计算

$$
g_i=\frac{\partial Q_{\mathrm{tot}}}{\partial q_i}.
$$

报告：

- $\min_i g_i$；
- 负梯度比例；
- $g_i$ 的 mean/median/max；
- 不同训练阶段的分布。

理论认证模型的负梯度比例应在数值容差内为 0。

### 8.2 Credit sensitivity

记录

$$
\|\nabla_{\mathbf q}Q_{\mathrm{tot}}\|_1,
\qquad
\min_i\frac{\partial Q_{\mathrm{tot}}}{\partial q_i},
\qquad
\max_i\frac{\partial Q_{\mathrm{tot}}}{\partial q_i}.
$$

这能够检验 HLL sigmoid 或 KAN tanh 是否饱和，以及某些 seed 是否因为 Q 梯度过弱而无法启动。

### 8.3 状态条件信用变化

核心机制不是二阶导数本身，而是状态改变时 Q-credit surface 是否改变。定义有限差分指标

$$
\mathcal I_{\mathrm{state-Q}}
=
\mathbb E\left[
\left\|
\nabla_{\mathbf q}M(\mathbf q,\mathbf z)
-
\nabla_{\mathbf q}M(\mathbf q,\mathbf z')
\right\|_1
\right],
$$

其中 $\mathbf z'$ 来自同一 batch 的随机置换状态。

解释：

- 接近 0：mixer 的 Q-credit 几乎不随状态变化；
- 较大：状态显著改变个体价值的组合方式；
- 过大且高方差：可能表示不稳定的过拟合。

应结合胜率和 TD error 分析，不能单独认为越大越好。

### 8.4 Shortcut 比例

若使用 residual，记录

$$
\rho_R
=
\frac{|R(\mathbf q)|}
{|M(\mathbf q,z)|+|V(z)|+|R(\mathbf q)|+\epsilon}.
$$

该指标可以验证 MMM2 失败 run 是否被 additive shortcut 主导。

### 8.5 训练稳定性

记录并汇总：

- TD loss；
- absolute TD error；
- gradient norm；
- Q taken / target mean；
- agent Q 的尺度；
- calibration 前后 Q 的饱和比例；
- NaN/Inf 和梯度裁剪频率。

---

## 9. 受控函数逼近实验

建议增加一个与 SMAC 分离的小型实验，直接验证“条件部分单调函数逼近”这一核心命题。该实验成本低，且比只看胜率更容易解释架构差异。

### 9.1 数据设置

采样

$$
\mathbf q\in[-1,1]^n,\qquad
\mathbf z\in[-1,1]^d.
$$

训练集和测试集独立生成，同时设置 context OOD 测试，例如训练 $z_j\in[-0.8,0.8]$，测试边界区域。

### 9.2 推荐目标函数

#### 状态条件线性权重

$$
f_1(\mathbf q,\mathbf z)
=b(\mathbf z)
+\sum_i\operatorname{softplus}(a_i(\mathbf z))q_i.
$$

用于检验状态改变 credit weights 的基本能力。

#### 状态条件阈值

$$
f_2(\mathbf q,\mathbf z)
=b(\mathbf z)
+\sum_i c_i(\mathbf z)\,
\sigma\big(\kappa(q_i-t_i(\mathbf z))\big),
$$

其中 $c_i(\mathbf z)\ge0$ 且 $\kappa>0$。因此每个 sigmoid 项关于对应的 $q_i$ 非递减。该函数用于检验 KAN/lattice 对局部阈值的建模能力。

#### 高阶协同

在 $q_i\in[0,1]$ 上定义

$$
f_3(\mathbf q,\mathbf z)
=b(\mathbf z)
+\sum_i a_i(\mathbf z)q_i
+c(\mathbf z)\prod_i q_i,
$$

其中 $a_i,c\ge0$。该函数对每个 $q_i$ 单调，但包含明显非加性交互。

#### 多模式单调曲面

$$
f_4(\mathbf q,\mathbf z)
=
\sum_{r=1}^{R}\rho_r(\mathbf z)f_r(\mathbf q),
\qquad
\rho_r(\mathbf z)\ge0,\quad
\sum_r\rho_r(\mathbf z)=1.
$$

其中每个 $f_r(\mathbf q)$ 对 $\mathbf q$ 逐坐标非递减。该目标直接对应理论文档中的有限混合逼近定理。

### 9.3 指标

- test MSE/MAE；
- monotonic violation rate；
- worst-case negative gradient；
- OOD context error；
- 参数量；
- 训练时间；
- 随 $n$ 和 $d$ 增长的误差。

该实验应同时比较 QMIX、PMIX-MLP、PMIX-Lattice、PMIX-KAN，并参数匹配。

---

## 10. 扩展性实验

使用不同 agent 数或不同 Q 输入维度，报告

$$
n\in\{3,5,8,10,20,27\}.
$$

可结合 SMAC 地图和合成函数实验。

重点报告：

- wall-clock/update；
- peak memory；
- mixer parameter count；
- HLL vertex count；
- forward/backward time；
- final/AUC performance。

PMIX-Lattice 必须明确 full lattice 和 grouped lattice 的适用范围。grouping 规则应由 agent 数和最大顶点预算确定，不能根据测试曲线人工选择。

---

## 11. 推荐的分阶段运行计划

### Phase A：代码和数值认证

目标：

- 三种 PMIX 统一 $E(s)$、$V(s)$ 和其他公共训练协议；
- 完整单元测试；
- 随机输入上的 forward/backward；
- 数值单调性检查；
- 参数量和吞吐统计。

不运行正式 SMAC 长实验。

### Phase B：合成函数与配置筛选

使用小型函数逼近实验筛选：

- state embedding dimension；
- base capacity；
- Q calibration；
- 初始化。

每种方法使用相同 trial 数。

### Phase C：开发地图调参

地图：

- 3s_vs_5z；
- MMM2。

使用 3 个开发 seeds 和固定搜索空间。选择标准使用两张地图 aggregate IQM/AUC，而不是单张地图 peak。

### Phase D：冻结主实验

冻结代码、配置和 seeds，在 4-6 张主地图上运行 5-8 seeds。

主实验开始后，不根据中间结果修改方法配置。发现代码 bug 时所有受影响方法统一重跑并记录版本。

### Phase E：机制消融

仅在 3s_vs_5z 和 MMM2 上运行：

- conditioning control；
- $V(s)$；
- residual；
- capacity。

PMIX-Lattice 的 output scale 属于架构内部消融，只在 Lattice 分支中比较。

### Phase F：扩展性和附录

运行 large-agent maps、额外简单地图和架构内部消融。

---

## 12. 主文图表建议

### Table 1：主结果

每个方法在每张主地图上的：

- final win rate；
- normalized AUC；
- 95% CI。

### Figure 1：学习曲线

展示代表性地图：

- 3s_vs_5z；
- 5m_vs_6m；
- MMM2；
- 27m_vs_30m。

### Figure 2：跨任务汇总

- IQM；
- performance profile；
- probability of improvement over QMIX。

### Table 2：公平性与效率

- 参数量；
- FLOPs 或 update time；
- peak memory；
- steps/second。

### Figure 3：机制分析

- $\nabla_{\mathbf q}Q_{\mathrm{tot}}$ 分布；
- state-Q interaction 指标；
- residual shortcut 比例。

### Table 3：核心消融

- Hyper-MLP vs Direct-MLP；
- with/without context interaction；
- no/fixed/annealed residual。

合成函数、完整地图结果和内部架构消融可放附录。

---

## 13. 现有结果应如何使用

当前实验已经支持以下假设：

1. HLL 的表现强烈依赖 agent 数和 lattice resolution；
2. Q residual 能改善部分地图的早期信用分配；
3. 固定 additive residual 在 MMM2 可能导致 VDN-like shortcut；
4. AMCO residual annealing 可能改善稳定性；
5. MonoKAN 对 residual、grid 和 temperature 较敏感；
6. 简单地图容易饱和，不能单独支撑表达力优势。

这些结果适合：

- 确定正式搜索范围；
- 设计 residual 消融；
- 选择代表性地图；
- 提出机制诊断指标。

但不适合直接作为最终主结论，因为存在：

- 不同方法调参轮数不同；
- map-specific 参数较多；
- seeds 数量有限；
- 多轮结果中选 best variant；
- 公共模块并不完全一致。

正式论文应将现有结果标为 pilot study，并在冻结协议后重跑最终对比。

---

## 14. 复现要求

服务器每个 run 保存：

- git commit hash；
- 完整 Sacred config；
- map、seed、SC2/SMAC 版本；
- CUDA/PyTorch 版本；
- 开始与结束时间；
- 参数量和 peak memory；
- 原始评估序列；
- checkpoint；
- 是否正常完成。

推荐结果目录包含实验协议版本，例如：

~~~text
results/
  pmix_main_v1/
  pmix_ablation_v1/
  pmix_scaling_v1/
~~~

禁止覆盖旧结果。绘图脚本应从 config 和 info 文件自动读取方法、地图和 seed，不依赖手工重命名。

## 15. 最小可发表实验包

如果计算预算有限，优先完成：

1. QMIX、VDN、三种 PMIX；
2. 3s_vs_5z、5m_vs_6m、MMM2、27m_vs_30m；
3. 每组至少 5 seeds；
4. 参数量匹配和公共模块统一；
5. Hyper-MLP vs Direct-MLP 核心对照；
6. 在 3s_vs_5z、MMM2 上做 residual 和 context-interaction 消融；
7. 报告 final、AUC、IQM、95% CI 和计算开销；
8. 一个小型条件部分单调合成函数实验。

如果这套实验成立，论文可以较有力地支持：

> PMIX 是一种保持 IGM、无需状态生成 mixer 参数、能够直接建模自由状态-Q交互的统一价值分解范式。
