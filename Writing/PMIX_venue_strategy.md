# PMIX 投稿 venue 分析与建议

本文基于当前 PMIX 的理论边界、已有相关工作、现有实验记录和
`Writing/PMIX_experimental_design.md` 中的最小可发表实验包制定。判断日期为
2026-07-21。会议截止日期和格式要求具有时效性，正式投稿前必须以对应年份的官方 CFP
为准。

## 1. 结论

当前最合适的第一目标是 **AAMAS 主会**，备选是 **AAAI 或 IJCAI**，期刊路线首选
**TMLR**，理论进一步加强后可考虑 **JAAMAS**。ICLR 是有吸引力的高上限目标，但需要
补强 direct conditioning 的理论和跨环境实验证据。NeurIPS/ICML 属于冲刺目标，不建议在
当前证据规模下作为唯一投稿出口。

推荐顺序如下：

1. AAMAS 主会：主题匹配度最高，能够自然接收 value factorisation、IGM、CTDE 和多智能体
   协作实验。
2. AAAI/IJCAI：适合将 PMIX 作为统一 AI 方法和实证框架介绍，但需要压缩理论并突出实际
   协作性能与可解释机制。
3. ICLR：适合把 PMIX 强调为一种新的受约束神经函数逼近范式；前提是理论和合成逼近实验
   达到机器学习论文标准。
4. TMLR：如果优先考虑技术完整性、可复现性和滚动投稿，TMLR 是最稳妥的期刊选择。
5. JAAMAS/JAIR：适合更长、更理论化的版本，但审稿周期通常更长，且需要显著扩展理论
   或多智能体分析。
6. NeurIPS/ICML：只有在补充严格的架构级逼近定理、明显的跨环境优势或新的学习理论后，
   才值得作为主要目标。

## 2. 当前工作的真实定位

PMIX 目前最有说服力的贡献组合是：

\[
Q_{\mathrm{tot}}=V(z)+M_{\phi}(\mathbf q,z),
\qquad
\frac{\partial M_{\phi}}{\partial q_i}\geq 0,
\]

其中对条件变量 \(z\) 不施加单调约束，并以 PMIX-MLP、PMIX-Lattice 和 PMIX-KAN
实现不同的部分单调归纳偏置。

这不是“首次发现状态可以和个体 Q 融合”，也不是“首次提出单调 mixer”。QMIX、Qatten、
NA²Q 等已有方法已经包含条件单调 mixing 的特例。PMIX 的可主张贡献应写成：

> PMIX provides an architecture-agnostic formulation of conditional partial-monotone
> value mixing, together with a unified IGM guarantee and multiple certified realizations
> beyond QMIX-style state-to-mixer-weight parameterization.

因此，投稿时必须把贡献重点放在“统一范式、认证接口、不同逼近器的归纳偏置和受控比较”，
而不能只把论文写成“将三个现有单调网络替换进 QMIX”。

## 3. Venue 对比

| Venue | 主题匹配 | 当前理论门槛 | 当前实验门槛 | 适合的论文叙事 | 建议 |
|---|---|---:|---:|---|---|
| AAMAS | 很高 | 中高 | 中高 | 多智能体 value factorisation 的统一范式 | 第一目标 |
| AAAI | 高 | 中 | 高 | 面向协作 MARL 的新方法和机制诊断 | 强备选 |
| IJCAI | 高 | 中高 | 高 | AI 方法、IGM 保证和跨任务实证 | 强备选 |
| ICLR | 中高 | 高 | 高 | 受约束神经函数逼近与 direct conditioning | 理论加强后冲刺 |
| TMLR | 高 | 高但可充分展开 | 中高 | 完整理论、复现协议和全面实验 | 最稳妥期刊 |
| JAAMAS | 很高 | 高 | 中高 | 多智能体理论和长期可复现分析 | 长文版本 |
| JAIR | 中高 | 很高 | 中 | 通用 IGM/函数逼近理论 | 需要更强定理 |
| NeurIPS | 中 | 很高 | 很高 | 新的普适受约束学习理论或显著性能突破 | 冲刺 |
| ICML | 中 | 很高 | 高 | 新的函数逼近理论、优化或泛化结论 | 冲刺 |

这里的“门槛”不是官方接受率，而是按照当前工作的贡献形态估计的证明、实验和叙事要求。

## 4. 为什么 AAMAS 最匹配

### 4.1 研究问题属于 AAMAS 的核心范围

PMIX 的核心对象是 cooperative MARL 中的 centralized value factorisation：个体历史、全局
上下文、IGM、CTDE 和 SMAC 协作任务都属于 AAMAS 主会读者熟悉的问题。

### 4.2 理论贡献不必被误解为纯数学网络论文

PMIX 的理论目标是证明：

\[
\text{conditional partial monotonicity}
\Longrightarrow
\text{pointwise IGM consistency}.
\]

这对 AAMAS 来说是清晰的多智能体保证；不需要把论文包装成一个脱离 MARL 的全新通用逼近
理论。三种架构认证命题也能作为方法设计的直接依据。

### 4.3 AAMAS 对“统一框架 + 系统实验”更友好

当前工作的强项不是单个网络在所有 SMAC 地图上都胜过 QMIX，而是：

- 三种不同结构共享同一数学接口；
- direct conditioning、value conditioning 和 hypernetwork conditioning 可以受控比较；
- 仍保持 IGM；
- 可以解释不同结构在小队、中等队伍和大规模队伍上的表达能力与扩展性差异。

这类多维方法论贡献比单一 benchmark 胜率更适合 AAMAS。

## 5. ICLR 需要补什么

如果选择 ICLR，论文主线应从“MARL mixer 替换”提高为“conditional partially monotone neural
approximation”。至少需要补齐：

1. 对 PMIX-MLP 给出 direct context pathway 的明确表示能力定理，而不仅是继承 AMCO 的
   全单调切片结论。
2. 对 PMIX-Lattice 给出网格细化下的误差界，说明 context-dependent vertex function 的
   一致逼近条件。
3. 对 PMIX-KAN 明确区分 spline 的一维逼近、全域单调认证和多变量部分单调函数的完整 UAT；
   不应把 MonoKAN 原论文结果直接升级成 PMIX-KAN UAT。
4. 增加合成条件单调函数实验，报告 uniform error、monotonicity violation、参数量和推理
   成本，而不只报告 SMAC 胜率。
5. 证明或实验验证 Direct-MLP 与 Hyper-MLP 在相同参数量和相同上下文编码下的差异。

没有这些补充时，ICLR 审稿人很可能认为 PMIX 只是把已知单调网络组合进 QMIX，而不是新的
函数逼近范式。

## 6. AAAI/IJCAI 需要补什么

如果以 AAAI 或 IJCAI 为目标，理论可以保持当前深度，但主文必须把实验做完整：

- QMIX、VDN、三种 PMIX，以及至少一个现代 IGM baseline；
- Hyper-MLP vs Direct-MLP；
- 4--6 个有难度梯度的 SMAC/SMACv2 地图；
- 每组至少 5 个正式 seeds，最好 8 个；
- final、AUC、IQM、bootstrap 95% CI 和计算开销；
- 单调性数值诊断和 IGM action-selection 检查；
- 一个小型合成条件部分单调函数实验。

AAAI/IJCAI 版本应避免过长的架构证明细节，把完整证明放到补充材料或附录；主文突出统一
定义、主定理、方法图、主结果和机制消融。

## 7. TMLR 与 JAAMAS 的选择

### TMLR

TMLR 最适合当前项目的原因是：

- 可以容纳完整的定义、证明、相关工作边界和实验协议；
- 允许明确说明哪些是 PMIX 原创推论，哪些继承自 HLL、AMCO 和 MonoKAN；
- 对复现材料、统计报告、代码和负面结果的要求与当前实验设计一致；
- 不必为了会议页数删除 HLL 扩展性限制、MIXRTs 单调性边界或合成函数实验。

如果正式主结果显示三种 PMIX 不是全面优于 QMIX，而是具有不同的任务归纳偏置，TMLR 比
只强调胜率的会议叙事更合适。

### JAAMAS

JAAMAS 适合在以下情况下考虑：

- 增加 stateful IGM、部分可观测历史和多智能体信用分配的理论讨论；
- 对大规模队伍的 lattice grouping、计算复杂度和可扩展性进行系统研究；
- 提供更全面的 SMAC、SMACv2、Google Research Football 或其他 cooperative MARL 环境；
- 论文篇幅需要完整保留理论和失败案例。

如果论文主要是神经网络架构和小规模 SMAC 实验，则 TMLR 更直接；如果论文要成为多智能体
价值分解的长期参考，JAAMAS 才值得投入。

## 8. 不建议当前直接投 NeurIPS/ICML 的原因

当前存在三个明显风险：

1. **函数类新颖性风险**：QMIX 已经是对 \(\mathbf q\) 单调、对状态条件化的函数。PMIX
   目前更像统一参数化和归纳偏置，而不是严格更大的函数类。
2. **逼近定理尚未架构闭合**：范式级有限混合逼近不能自动推出三个具体计算图都具有完整 UAT。
3. **实验规模尚未达到顶级 ML 会议要求**：现有历史结果主要是探索性 3-seed 结果，且 PMIX
   尚未在所有地图上稳定超过 QMIX。

只有在以下结果出现后，NeurIPS/ICML 才值得作为主要目标：

- 一个非平凡且严格的新逼近或泛化定理；
- 多个环境中 direct conditioning 的稳定收益；
- 参数匹配后仍然显著优于 QMIX/Hyper-MLP；
- 清晰的 scaling 或 sample-efficiency law；
- 对失败地图和结构限制有可验证解释。

## 9. 建议的投稿决策门

### Gate A：当前探索结果

如果仍只有 3 seeds、SMAC 主实验和部分架构认证：不要投稿顶级 ML 主会。先完成代码冻结、
正式 seeds 和 Hyper-MLP/Direct-MLP 对照。

### Gate B：最小完整论文

若达到 5--8 seeds、4--6 张地图、合成函数实验、完整单调诊断和公平参数匹配：

- 首投 AAMAS；
- 同时准备 AAAI/IJCAI 风格的压缩版本；
- 若希望避免固定会议周期，则转投 TMLR。

### Gate C：理论增强版本

若补齐三个架构的逼近边界，并证明 direct context 带来明确的表示或样本效率优势：

- ICLR 可以作为第一目标；
- AAMAS 作为领域匹配的稳健备选；
- TMLR 作为完整长文出口。

## 10. 最终推荐

当前最合理的策略是：

> **主目标：AAMAS。** 论文定位为 cooperative MARL 的统一 conditional partial-monotone
> value mixing 范式，重点展示 IGM 认证、三种架构和受控实验。

> **理论增强后：ICLR。** 将论文重写为受约束条件函数逼近工作，并补齐架构级逼近结论。

> **稳妥期刊：TMLR。** 如果实验与理论完整，但不形成顶会级别的单一突破，TMLR 是最自然
> 的长期版本。

> **长期理论版本：JAAMAS/JAIR。** 只有在 stateful IGM、可扩展性和函数类理论明显扩展后
> 才值得选择。

不建议在摘要中使用“first partial monotonic mixer”。建议使用：

> To the best of our knowledge, PMIX is the first unified architecture-agnostic treatment
> of conditional partial-monotone value mixing with certified MLP, lattice, and KAN
> realizations in cooperative MARL.

