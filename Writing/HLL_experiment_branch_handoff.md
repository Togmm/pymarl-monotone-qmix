# HLL 实验分支交接总结

更新时间：2026-08-06

本文用于接替后续 HLL/PMIX 对话，记录当前代码分支、已经完成的实验、结果解释和下一步边界。实验长跑应在服务器进行；本地仓库只保留代码、配置、日志快照和分析文档。

## 1. 当前状态

仓库：`/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl`

当前分支：`master`

当前 HLL 代码提交：`5bddc0e`（提交说明为 `1`）。工作区已有一个与本文件无关的未跟踪文件 `Writing/AMCO_experiment_handoff.md`，不要覆盖或删除。

核心文件：

- `src/modules/mixers/hll_monotone.py`
- `src/config/algs/hll.yaml`
- `src/learners/q_learner.py`
- `run_parallel_pymarl_exps.sh`
- `tests/test_mixers.py`

当前分支的研究问题是：HLL 的困难主要来自 Q 输入经过固定 sigmoid 后的尺度/饱和问题，还是来自 lattice 分辨率和状态条件顶点函数本身？因此当前采用单因素的 learned monotone Q calibrator，而没有同时改动 lattice、residual、endpoint normalization 或 output scale。

## 2. HLL 当前形式

### 2.1 标准 HLL 输出

当前 HLL 可以概括为：

$$
Q_{\mathrm{tot}} = V(z) + A(z)\left[L(c_1,\ldots,c_g;z)-\frac{1}{2}\right] + R_Q.
$$

其中：

- `z` 是状态输入；
- `V(z)` 是不参与动作排序的状态价值分支；
- `L(c;z)` 是状态条件 lattice；状态辅助网络根据 `z` 生成有序顶点值；
- `A(z)>0` 是 HLL 内部的正输出尺度，用于把有界、中心化的 lattice 输出恢复到联合价值范围；
- `R_Q` 是代码中保留的可选 Q residual，但当前标准配置没有该参数，因此代码默认 `q_residual_scale=0`，实际标准实验不使用 residual。

HLL 的 monotone 坐标数量是 `g`。通常没有 grouping 时 `g=n_agents`；`27m_vs_30m` 和 `bane_vs_bane` 使用分组以控制完整 lattice 的指数增长。

### 2.2 Q 输入校准

关闭 calibrator 时，代码使用旧的固定变换：

$$
c_i = \sigma\left(\frac{Q_i}{T}\right).
$$

开启 learned calibrator 时，代码使用：

$$
c_i = \sigma\left(\frac{Q_i-b}{\tau}\right),
\qquad
\tau=\operatorname{softplus}(\rho)+\tau_{\min}>0.
$$

其中 `b` 和 `rho` 是跨 agent/group 共享的可学习参数，当前配置为：

```yaml
hll_q_calibrator_enabled: True
hll_q_calibrator_init_shift: 0.0
hll_q_calibrator_init_scale: 1.0
hll_q_calibrator_min_scale: 0.25
```

它没有状态输入，不是 hypernetwork。由于 `tau>0`，校准器严格单调，不改变 HLL 对个体 Q 的 IGM 单调性。`hll_q_temperature_by_map` 全部恢复为 `1.0`，只在 calibrator 关闭时作为 fallback。

### 2.3 当前 lattice 和 grouping 规则

当前 `hll.yaml` 的 map-specific resolution 如下：

| 地图 | 己方 agent | lattice resolution | 近似顶点数 | grouping |
|---|---:|---:|---:|---|
| `MMM2` | 10 | `2` | `2^10=1024` | 无 |
| `3s_vs_5z` | 3 | `6` | `6^3=216` | 无 |
| `5m_vs_6m` | 5 | `4` | `4^5=1024` | 无 |
| `3s5z` | 8 | `[3,3,3,2,2,2,2,2]` | `864` | 无 |
| `1c3s5z` | 9 | `[3,3,3,3,2,2,2,2,2]` | `2592` | 无 |
| `2s3z` | 5 | `4` | `1024` | 无 |
| `2c_vs_64zg` | 2 | `16` | `256` | 无 |
| `bane_vs_bane` | 24 | `2` | `2^10=1024` | 10 groups, contiguous |
| `27m_vs_30m` | 27 | `2` | `2^10=1024` | 10 groups, sorted |

`hll_max_vertices=4096`。HLL 的顶点数随输入维度指数增长，因此 `size` 不是跨地图等容量：例如 `MMM2` 的 binary lattice 有 1024 个顶点，而低维地图需要更大的 per-dimension resolution 才能得到相近的分辨率。

## 3. 实验阶段和代码变化

### 3.1 初始 HLL 与 map-aware lattice

早期实验表明，统一使用 `size=2` 并不等价于统一容量：

- `MMM2` 有 10 个 agent，`2^10=1024` 个顶点，binary lattice 仍有较大的联合组合空间；
- `3s_vs_5z` 只有 3 个 agent，`2^3=8` 个顶点，表达能力明显不足；
- 因此 `3s_vs_5z` 的 native 配置改为 `size=6`，而不是继续给所有地图使用 `size=2`。

早期 HLL 改进曾使用 Q residual 和提高 temperature。它们在特定地图上可以提供更强的早期梯度，但会改变原始 HLL 的归纳偏置：尤其是多 agent 地图上，固定 residual 容易形成 VDN-like shortcut。因此标准 PMIX 范式实验去掉了 residual。

### 3.2 HLL 诊断分支

在不改变正常输出的前提下，加入了以下诊断：

- `credit_grad_*`：`dQtot/dQi` 的均值、分位数、近零比例、负值比例、CV 和 credit entropy；
- `hll_q_*`：Q 坐标均值、低端/高端 saturation 和逐 agent/group 的 saturation；
- `hll_sigmoid_sensitivity_*`：sigmoid 坐标对原始 Q 的敏感度；
- `hll_output_scale_*`：`A(z)` 的动态范围；
- `hll_v_to_m_ratio`：状态价值分支与 lattice mixing 分支的幅值比；
- `hll_vertex_delta_*`：相邻 lattice 顶点的平均斜率、近零斜率和低/中/高层斜率；
- `hll_aux_probability_*`：顶点生成网络 sigmoid 输出的饱和情况。

诊断拓扑没有加入 `state_dict`，以免破坏旧 checkpoint 的加载。输出位置为：

- Sacred：`results_Pradigm_v2/sacred/hll/<run_id>/info.json`；
- TensorBoard：`results_Pradigm_v2/tb_logs/<experiment>__<timestamp>/`；
- 旧的并行运行文本：`parallel_logs/hll_*.log`。

在这些 Sacred 运行中，`metrics.json` 可能为空，实际时间序列在 `info.json` 中；例如胜率使用 `test_battle_won_mean` 和 `test_battle_won_mean_T`。

### 3.3 当前 learned-calibrator 分支

提交 `5bddc0e` 做了三件事：

1. 在 `hll_monotone.py` 中加入 `LearnedMonotoneQCalibrator`；
2. 在 `hll.yaml` 中打开 calibrator，并将所有 fallback temperature 恢复为 `1.0`；
3. 将 `run_parallel_pymarl_exps.sh` 从 `MMM2 T=2` 和 `3s_vs_5z K=2` 两个干预改成 native lattice + learned calibrator。

当前脚本默认使用 seeds `1, 41, 141`、`T_MAX=2050000`，运行两张地图，共 6 个任务：

- `hll_calibrated_MMM2`
- `hll_calibrated_3s_vs_5z`

本地没有发现 `hll_calibrated_*` 的 Sacred、TensorBoard 或并行日志目录。因此截至本文更新时间，learned-calibrator 的正式训练结果尚未同步到本地，不能把它写成已经有效或无效。

## 4. 标准 HLL 诊断实验结果

下面的主表来自 `results_Pradigm_v2/sacred/hll/*/info.json`。`Final` 是最后一个测试点，`Last10` 是最后 10 个测试点均值，`AUC` 是按测试时间归一化的胜率积分。种子顺序统一为 `1 / 41 / 141`。

### 4.1 Native resolution、temperature=1

| 地图 | Final | Last10 | AUC | 结果概括 |
|---|---|---|---|---|
| `MMM2`, `K=2` | `0.938 / 0.562 / 0.094` | `0.928 / 0.681 / 0.059` | `0.430 / 0.221 / 0.004` | seed141 几乎失败，seed 间差异极大 |
| `3s_vs_5z`, `K=6` | `0.344 / 0.875 / 0.594` | `0.394 / 0.859 / 0.516` | `0.080 / 0.250 / 0.113` | 没有普遍 Q saturation，但学习速度/最终效果不稳 |
| `5m_vs_6m`, `K=4` | `0.750 / 0.750 / 0.625` | `0.650 / 0.759 / 0.631` | `0.451 / 0.531 / 0.379` | 中等难度，三 seed 都能学习，仍有提升空间 |
| `2c_vs_64zg`, `K=16` | `0.938 / 0.969 / 0.938` | `0.866 / 0.925 / 0.941` | `0.625 / 0.496 / 0.411` | 整体较强，Q calibration 不是主要瓶颈 |

### 4.2 MMM2 的 Q temperature=2 对照

这组实验只把 `MMM2` 的 temperature 从 `1` 改成 `2`，保持 `K=2` 和其他设置不变：

| seed | Final | Last10 | AUC | 诊断 |
|---:|---:|---:|---:|---|
| 1 | `0.656` | `0.666` | `0.271` | saturation 消失，但性能低于 native seed1 |
| 41 | `0.000` | `0.000` | `0.000` | saturation 消失，仍然完全失败 |
| 141 | `0.438` | `0.484` | `0.102` | 相比 native seed141 有恢复，但仍不稳定 |

并行日志最后诊断快照显示，T=1 的 MMM2 seed141 曾出现：

- `hll_q_saturation_frac=0.5856`；
- `hll_sigmoid_sensitivity_mean=0.0744`；
- `credit_grad_mean=0.0400`；
- `hll_v_to_m_ratio=1.6713`；
- `hll_output_scale_mean=2.9056`。

T=2 确实把 saturation 降为 0，但同时降低了 sigmoid sensitivity，而且不同 seed 的 output scale、V/M 比和 Q credit 仍然差异很大。因此“提高 temperature”只能缓解一个表面症状，不能作为稳定的通用方案。

### 4.3 `3s_vs_5z` 的 K=2 对照

这组实验只将 native `K=6` 改成 `K=2`：

| seed | Final | Last10 | AUC | 诊断 |
|---:|---:|---:|---:|---|
| 1 | `0.625` | `0.787` | `0.220` | 反而比 native seed1 好 |
| 41 | `0.000` | `0.022` | `0.001` | 高端 Q saturation 约 `0.1484`，几乎失败 |
| 141 | `0.625` | `0.478` | `0.102` | 没有明显 saturation，但没有稳定提升 |

因此，K=2 不是一个可接受的统一替代。它改变了顶点斜率分布和 auxiliary 输出饱和，而不是单纯减少表达能力；seed41 的失败说明 capacity、Q calibration 和优化轨迹存在耦合。

## 5. 诊断结论

### 5.1 MMM2

MMM2 的核心现象不是顶点值没有状态依赖。binary HLL 仍由状态生成 1024 个 ordered vertices，且 `V(z)` 和 `A(z)` 都保留状态输入。

更可信的解释是：

1. 某些 seed 的 Q 输入进入 sigmoid 饱和区；
2. Q 到 lattice 的有效 sensitivity 变小；
3. state value 分支可能相对过强，导致 `Qtot` 能拟合状态回报但个体 Q 的 action-dependent credit 不足；
4. 单独把 temperature 调大并不能控制 output scale、V/M 比和 lattice slope，因此不能稳定恢复训练。

### 5.2 `3s_vs_5z`

native `K=6` 已经避免了 `2^3=8` 顶点的明显容量瓶颈，三个 seed 的 Q saturation 基本为 0，sigmoid sensitivity 约为 `0.205-0.232`。但胜率仍然跨 seed 波动，说明它的瓶颈可能还包括：

- state-conditioned vertex generator 的优化；
- 不同 lattice 层的 slope 分布；
- state value 与 mixing branch 的训练平衡；
- TD 学习的随机性。

因此，不能把 `3s_vs_5z` 的失败全部归因于 Q saturation，也不能把更大的 K 直接等同于更好的 HLL。

### 5.3 `5m_vs_6m`

这是当前最适合作为 calibrator 第三张地图的历史基线：五个同质 Marine、无 grouping、`4^5=1024` 个顶点，和 MMM2 具有相同的顶点预算但不同的输入维度。它的 native HLL 结果约为 `0.625-0.750` Final，具有足够 headroom，也没有 MMM2 那样的极端 seed141 崩溃。

5m 的旧诊断快照大致显示：Q saturation 约 `0.010-0.023`，sigmoid sensitivity 约 `0.196-0.205`，但 vertex near-zero slope 约 `0.104-0.129`。这张地图适合判断 learned calibrator 是否能改善 credit 曲面，而不把异质兵种或 grouping 引入为额外混杂因素。

### 5.4 其他地图

- `2c_vs_64zg` 的 native 结果已经较高，且只有两个 Q 输入，Q calibration 不是当前最明显的瓶颈；适合作为强状态条件 sanity check，而不是首个 calibrator 增量地图。
- `3s5z` 的 HLL 历史结果大致在 `0.875-0.938`，适合之后检验异质多 agent credit，但 shared calibrator 是否能覆盖不同兵种的 Q 尺度会形成额外问题。
- `2s3z`、`1c3s5z` 和 `bane_vs_bane` 很容易接近饱和，不适合支撑 calibrator 的性能结论。
- `27m_vs_30m` 使用 10-group sorted grouping，历史 HLL AUC 很低。它同时考验 grouping、agent identity 丢失和大规模扩展，不能作为当前 calibrator 的首个因果验证地图。

## 6. 当前 learned-calibrator 实验协议

当前唯一改变是：

```text
fixed sigmoid(Q / T)
        -> shared learned sigmoid((Q - b) / tau)
```

其余内容必须保持不变：

- 使用 native map lattice resolution；
- temperature fallback 保持 `1.0`；
- `q_residual_scale=0`；
- 保留 HLL 原有 `V(z)`、state-conditioned vertices 和 positive `A(z)`；
- 不同时加入 endpoint normalization、per-agent calibrator、residual、annealing 或 lattice-size 改动；
- 运行相同 seeds 和 `T_MAX=2050000`。

当前脚本默认启动：

```bash
cd /Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl
conda activate pymarl

CUDA_DEVICES="0 1 2 3 4 5" \
SEEDS="1 41 141" \
T_MAX=2050000 \
USE_TENSORBOARD=True \
bash run_parallel_pymarl_exps.sh
```

如果增加第三张地图，建议是 `5m_vs_6m`，但当前脚本尚未加入该组；新增后应仍使用相同的 seeds 和配置，不要只给新地图增加 seed 或改变 lattice。

## 7. 需要读取的指标

learned calibrator 运行完成后，首先比较以下轨迹，而不是只看最终胜率：

1. `hll_q_calibrator_shift`：共享中心是否从 0 移动；
2. `hll_q_calibrator_scale`：有效温度是否自适应增大或减小；
3. `hll_q_low_saturation_frac`、`hll_q_high_saturation_frac` 和 `hll_sigmoid_sensitivity_mean`；
4. `credit_grad_mean`、`credit_grad_p10`、`credit_grad_near_zero_frac` 和 `credit_entropy`；
5. `hll_vertex_delta_near_zero_frac` 以及 low/mid/high-level slope；
6. `hll_v_to_m_ratio`、`hll_output_scale_mean` 和 `hll_mixing_output_rms`；
7. test win rate 的 AUC、Last10、最终值和跨 seed 方差。

支持 calibrator 的机制证据应同时满足：saturation 降低或保持低位、有效 sensitivity 不塌陷、近零 credit 比例下降或不恶化、V/M 比没有异常偏向，并且 AUC/seed 稳定性改善。仅有最终胜率提升，不能证明 calibrator 解决了 Q credit 问题。

## 8. 未解决问题和禁止提前下结论的事项

### 8.1 尚未有 learned-calibrator 训练结果

当前本地只确认代码、配置、脚本和单元测试变更，没有确认 `hll_calibrated_*` 的服务器运行结果。因此以下说法目前都不能成立：

- calibrator 已经提高 HLL 性能；
- calibrator 已经消除了 MMM2 seed variance；
- calibrator 在所有地图上都优于 fixed sigmoid；
- HLL 的主要瓶颈已经被证明是 Q saturation。

这些只能作为待验证假设。

### 8.2 5m 旧日志中的负梯度比例

部分旧 HLL 文本日志记录到 `credit_grad_negative_frac` 约 `0.0046-0.0063`。理论上当前 HLL 的 Q 单调性应使该值在数值容差内为 0。由于这些日志与不同结果目录存在重复命名，且可能不是 learned-calibrator 分支，正式结论前应：

- 用当前代码和当前配置重跑一个短 smoke run；
- 确认 `q_residual_scale=0`；
- 区分真正的负梯度、浮点误差和日志覆盖；
- 再决定是否需要修复诊断或 HLL 插值实现。

### 8.3 shared calibrator 的异质性限制

当前 `b` 和 `tau` 对所有 agents/groups 共享。这是为了保持最小修改和公平性。若 `3s5z` 或 `MMM2` 的不同兵种 Q 分布明显不同，shared calibrator 可能无法同时校准所有坐标。但现在不能直接增加 per-agent/per-type calibrator，否则会把“验证 shared calibrator”变成另一个架构实验。

### 8.4 endpoint normalization 必须单独测试

“保留 vertex generator + 状态内端点归一化”是另一个 HLL 内部假设，不能和 learned calibrator 同时打开。正确顺序是：先完成 fixed sigmoid vs learned calibrator 的 paired comparison，再把 endpoint normalization 作为 HLL-only one-factor ablation。

## 9. 既有验证记录

此前在有 PyTorch 的 `epymarl` 环境中，HLL/diagnostic/calibrator 相关单元测试通过；已覆盖：

- HLL forward 开关诊断前后的输出一致性；
- low/high Q saturation 指标；
- grouped HLL 对原始 agent Q 的单调性；
- planned map lattice sizes；
- learned calibrator 的单调性、非负 Q 梯度、初始化 shift/scale 和有限值诊断。

本地 `pymarl` 环境曾出现 `ModuleNotFoundError: No module named 'torch'`，所以不要把本地环境的测试失败误认为代码逻辑失败，也不要在本地启动正式 SC2 长跑。

## 10. 一句话结论

当前 HLL 证据支持的最稳妥结论是：HLL 的表现同时受 lattice resolution、Q 输入校准、state/mixing 分支平衡和 seed 优化轨迹影响；temperature 或 lattice size 单独调整都不能稳定解决问题。当前最干净的下一步是，在 native HLL 配置上只启用 shared learned monotone Q calibrator，先比较 `MMM2` 和 `3s_vs_5z`，再用 `5m_vs_6m` 做中等维度、无 grouping 的第三张地图验证。

