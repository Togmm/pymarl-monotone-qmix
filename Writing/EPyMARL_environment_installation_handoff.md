# EPyMARL 多环境安装接续摘要

更新时间：2026-08-07

本文用于后续对话接续。它只总结当前研究背景、环境选择、框架支持状态和安装前必须确认的问题，不在当前阶段实现环境安装、GRF wrapper 或 PMIX 迁移功能。

## 1. 当前研究背景

当前研究方向是 PMIX（Partial Monotonic Mixing），即使用条件部分单调函数

$$
Q_{\mathrm{tot}} = F(Q_1,\ldots,Q_n,z)
$$

替代 QMIX 中基于 hypernetwork 的状态条件参数生成。PMIX 要求：

$$
\frac{\partial Q_{\mathrm{tot}}}{\partial Q_i} \ge 0,
\qquad i=1,\ldots,n,
$$

但不限制状态或历史编码 $z$ 对混合函数的影响方式，从而在保持 IGM 所需单调性的同时，让状态直接参与个体价值之间的交互。

当前主要变式包括：

- PMIX-MLP，对应 AMCO 路线；
- PMIX-Lattice，对应 HLL 路线；
- PMIX-KAN，对应 MonoKAN 路线。

正式实验遵循统一范式：不使用 Q residual，不给某个变式单独添加 VDN auxiliary loss、退火 credit shortcut 或其他只服务于单一架构的机制。不同环境中的比较也应保持这一原则。

## 2. 最终选定的环境

最终确定使用以下四类环境：

1. SMACv1；
2. SMACv2；
3. Google Research Football（GRF）；
4. Level-Based Foraging（LBF）。

它们在论文中的角色并不相同。

| 环境 | 在实验中的主要角色 | 重点考察 |
|---|---|---|
| SMACv1 | 与已有 PyMARL 结果衔接的主基线 | 战术协作、异质 agent、已有结果可比性 |
| SMACv2 | StarCraft 域内泛化验证 | 随机出生位置、随机兵种、状态条件泛化 |
| LBF | 低成本的外部离散协作环境 | 稀疏奖励、agent 等级、合作对象选择、credit assignment |
| GRF | 更复杂的外部协作环境 | 角色分工、传球配合、长期延迟 credit、状态变化 |

## 3. 选择这些环境的原因

### 3.1 SMACv1

SMACv1 是当前项目所有历史实验的基础。AMCO、HLL、MonoKAN、QMIX 和 VDN 的现有曲线、调参记录与诊断指标都来自 SMACv1，因此它必须保留。

它的主要作用不是提供新的环境多样性，而是：

- 验证迁移到 EPyMARL 后是否保持原有训练行为；
- 为 PMIX 与 QMIX 的主要性能比较提供连续证据；
- 继续分析不同兵种、agent 数量和 grouping 对 credit 分配的影响。

### 3.2 SMACv2

SMACv2 与 SMACv1 使用相近接口，但增加了更强的程序化随机性，主要包括：

- 随机出生位置；
- 随机单位类型；
- 调整 sight range 和 attack range；
- 同一场景在不同 episode 中出现不同的状态和队伍组成。

因此 SMACv2 适合检验 PMIX 的核心假设：当状态对合理 credit 分配方式具有更强影响时，直接的状态与个体价值交互是否比 hypernetwork 参数生成更有效。

SMACv2 仍然属于 StarCraft 环境，所以它是域内泛化证据，不能单独代替非 StarCraft 环境验证。

### 3.3 LBF

LBF 是当前最推荐优先安装和验证的非 StarCraft 环境，原因包括：

- 离散动作，与当前 QMIX/PMIX 的 value-based 学习方式兼容；
- agent 和食物具有等级，合作是否成功取决于参与 agent 的组合；
- 三个及以上 agent 时，需要决定与谁合作以及何时独立收集；
- 奖励较稀疏，但仿真成本明显低于 SMAC 和 GRF；
- 官方明确提供 cooperative 和 shared-reward 设置，适合研究 credit assignment。

LBF 的价值在于以较低成本判断 PMIX 是否真正改善状态相关的个体贡献分配，而不只是适应 StarCraft 的特殊状态表示。

### 3.4 GRF

GRF 提供更长时程和更具语义的协作问题，例如：

- 不同球员承担不同位置和职责；
- 传球与跑位需要跨多个时间步产生收益；
- 得分奖励高度延迟；
- 同一个局面下，状态会改变不同球员动作的团队价值。

这些特点与 PMIX 所关注的“状态如何改变个体价值之间的组合关系”高度相关。但 GRF 的工程成本最高，因此应在 SMACv1、SMACv2 和 LBF 的 EPyMARL 基线跑通之后再处理。

## 4. 当前框架支持状态

### 4.1 当前 PyMARL checkout

当前项目中的 PyMARL 只原生支持 SMACv1：

- `src/envs/__init__.py` 只有 `REGISTRY["sc2"]`；
- `src/config/envs/` 只有 `sc2.yaml` 和 `sc2_beta.yaml`；
- `requirements.txt` 安装的是原始 `smac`；
- `sc2_beta.yaml` 不是 SMACv2 配置。

因此当前 PyMARL 不适合作为四环境统一实验框架。

### 4.2 EPyMARL

截至本次核对，EPyMARL 的支持情况如下：

| 环境 | EPyMARL 状态 | 对应入口 |
|---|---|---|
| SMACv1 | 原生支持 | `sc2` |
| SMACv2 | 原生支持 | `sc2v2` |
| LBF | 官方支持，通过通用 Gymnasium wrapper | `gymma` |
| GRF | 没有原生环境注册 | 后续需要专用 wrapper |

EPyMARL 官方仓库提供：

- `src/config/envs/sc2.yaml`；
- `src/config/envs/sc2v2.yaml`；
- `src/config/envs/smacv2_configs/`；
- `src/config/envs/gymma.yaml`；
- SMACv1、SMACv2 和 LBF 的运行示例。

因此，后续建议使用 EPyMARL 作为统一代码基线，而不是继续扩展当前 PyMARL 的单环境注册。

## 5. 安装前必须注意的问题

### 5.1 固定 EPyMARL 版本

EPyMARL 的多数 Python 依赖没有严格固定版本。后续安装时必须记录：

- EPyMARL commit 或 release tag；
- Python 版本；
- PyTorch 和 CUDA 版本；
- 完整依赖快照；
- 服务器操作系统和 GPU 驱动信息。

不能使用浮动的 `main` 完成一部分实验，再升级依赖完成另一部分实验。

本次核对的 EPyMARL `main` 提交为：

```text
cbc38c09588064eab978501d0f12c2cf58fa7fc2
```

该提交可作为后续安装对话的候选起点，但安装前仍应再次确认服务器准备使用的版本。

### 5.2 SMACv1 与 SMACv2 的 SC2 版本

这是最重要的可比性问题：

- SMACv1 论文结果使用 `SC2.4.6.2.69232`；
- SMAC 官方明确说明不同 SC2 版本的性能不能直接比较；
- SMACv2 官方推荐的安装脚本使用 `SC2.4.10`。

若四环境正式实验统一使用 EPyMARL，当前更合理的方案是使用 SMACv2 推荐的 SC2 版本，并在 EPyMARL 中重新运行需要进入正式比较表的 SMACv1 baseline。旧 PyMARL 曲线应保留为调参和机制分析历史，不能在未核对 SC2 版本时直接并入新结果。

安装前需要先检查原服务器历史实验实际使用的：

- `SC2PATH`；
- `$SC2PATH/Versions/`；
- SMAC maps 版本；
- Sacred 配置或日志中是否保存了 SC2 版本信息。

### 5.3 LBF 的奖励语义

LBF 原生可返回逐 agent 奖励，但 QMIX、VDN、QTRAN 和当前 PMIX 属于共同奖励方法。正式实验必须统一使用共同奖励，并固定 reward scalarisation。

需要避免以下不公平情况：

- 一部分模型使用 individual rewards，另一部分模型使用 common reward；
- 一部分实验对奖励求和，另一部分求平均；
- 不同 LBF 任务混用 cooperative 和非 cooperative 规则，却仍作为同一组结果比较。

第一轮 LBF 建议优先考虑三个 agent 的任务，使 credit 分配具有足够区分度，而不是只使用两 agent 简单任务。

### 5.4 GRF 不是安装后即可用于 EPyMARL

安装 `gfootball` 只能证明 GRF 环境本身可用，不能证明 EPyMARL 已经能够训练。

GRF 与 EPyMARL 之间还存在以下接口差异：

- GRF 使用 Gym 风格 API，EPyMARL 当前通用 wrapper 面向 Gymnasium；
- GRF 多球员模式要求调用方传入联合动作并拆分观测；
- EPyMARL 需要固定 `n_agents`、observation size、state size 和 action space；
- GRF 原生 `get_state()` 返回用于恢复模拟器的 opaque state，不是可以直接输入 mixer 的数值全局状态；
- 共同奖励、reward shaping、动作集合和 episode limit 都需要显式定义。

因此后续工作应分成两个阶段：

1. GRF 环境安装与原生多球员 smoke test；
2. 单独讨论并实现 EPyMARL GRF wrapper。

当前交接文档不提前实现第二阶段。

## 6. 推荐的后续推进顺序

环境安装和 PMIX 迁移应按以下顺序推进：

1. 建立版本固定的 EPyMARL 环境；
2. 验证 EPyMARL 原始 QMIX；
3. 验证 LBF；
4. 验证 SMACv1；
5. 验证 SMACv2；
6. 安装并验证原生 GRF；
7. 讨论 GRF wrapper 的状态、动作和奖励设计；
8. 四个环境的 QMIX 基线链路确认后，再迁移 PMIX；
9. 先做短 smoke test，再开始服务器正式多 seed 实验。

环境适配阶段不要同时改动 PMIX 架构、激活函数、residual、lattice size 或其他模型超参数，否则无法区分问题来自环境接口还是 mixer。

## 7. 后续统一实验需要保持的条件

跨环境比较至少应统一：

- PMIX 的形式化范式；
- 是否包含 $V(z)$；
- 不使用 Q residual；
- agent network 和 learner 主体；
- common reward 定义；
- 每个环境内部的训练步数、测试间隔和 seeds；
- QMIX、VDN 与各 PMIX 变式的输入信息；
- state 构造方式；
- 诊断指标的记录频率和定义。

不同环境可以有不同的网络宽度或训练预算，但这些差异必须作为环境级配置公开，不能只为某个 PMIX 变式单独设置。

## 8. 下一次安装对话需要提供的信息

下一次正式开始安装前，需要先确认：

1. 服务器操作系统和是否具有 `sudo` 权限；
2. 是否支持 Docker、Singularity 或 Apptainer；
3. GPU、驱动和 CUDA 版本；
4. 准备安装 EPyMARL 的服务器路径；
5. 希望沿用的 conda 环境名称；
6. 服务器当前是否已经安装 StarCraft II；
7. 现有 `SC2PATH` 和 `Versions/` 内容；
8. 现有 SMAC maps 路径；
9. 是否希望复现 SMACv1 的 4.6.2 环境，还是统一迁移到 4.10；
10. GRF 是否允许使用容器安装系统依赖。

建议使用以下提示词接续：

```text
请先读取 Writing/EPyMARL_environment_installation_handoff.md。
我现在准备在服务器安装 EPyMARL，服务器信息如下：
- 系统：...
- GPU/CUDA：...
- 是否有 sudo/容器权限：...
- 目标安装路径：...
- 当前 SC2PATH 和 SC2 版本：...
- conda 环境：...
请从版本冻结和基础环境检查开始，一次只完成一个安装阶段。
```

## 9. 官方资料

- EPyMARL：<https://github.com/uoe-agents/epymarl>
- EPyMARL 环境依赖：<https://github.com/uoe-agents/epymarl/blob/main/env_requirements.txt>
- EPyMARL 环境注册：<https://github.com/uoe-agents/epymarl/blob/main/src/envs/__init__.py>
- SMACv1：<https://github.com/oxwhirl/smac>
- SMACv2：<https://github.com/oxwhirl/smacv2>
- SMACv2 maps：<https://github.com/oxwhirl/smacv2/releases/tag/maps>
- LBF：<https://github.com/uoe-agents/lb-foraging>
- GRF：<https://github.com/google-research/football>
- GRF 多智能体接口：<https://github.com/google-research/football/blob/master/gfootball/doc/multi_agent.md>
- GRF 环境 API：<https://github.com/google-research/football/blob/master/gfootball/doc/api.md>

## 10. 当前状态

- 四个目标环境已经确定；
- EPyMARL 是后续统一框架；
- 当前 PyMARL 不支持 SMACv2、LBF 或 GRF；
- EPyMARL 原生支持 SMACv1、SMACv2 和 LBF；
- GRF 需要在环境安装后另行适配；
- 当前尚未在服务器执行新的环境安装；
- 当前尚未向 EPyMARL 迁移 PMIX 代码；
- 当前文档只用于交接，不代表任何环境已安装或验证成功。
