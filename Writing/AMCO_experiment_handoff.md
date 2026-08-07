# AMCO 实验分支交接摘要

更新时间：2026-08-06

本文用于后续对话接替 AMCO 实验。它记录当前代码、统一配置、激活函数筛选、诊断指标、已完成结果和待运行任务。正式长时间训练仍在服务器的 `conda activate pymarl` 环境中执行，本地只保留代码、日志和分析材料。

## 1. 当前结论

当前 AMCO 的统一候选配置为：

- 单调分支激活函数：centered Softplus，`beta=2.0`；
- 状态输入缩放：所有地图统一 `amco_state_input_scale=0.3`；
- Q residual：关闭，`q_residual_scale` 不在配置中，代码 fallback 为 `0.0`；
- 状态值分支保留为 `V(s)`，但不承担 action-dependent credit；
- 不使用 VDN auxiliary loss、credit floor 或其他人为均匀 credit 通路；
- 继续保留诊断指标，但诊断不改变 mixer 前向、TD target 或随机数状态。

目前的判断是：centered Softplus beta=2 在样本效率和跨地图稳定性上是最有希望的统一选择，但它并没有消除地图相关的 seed 方差。`5m_vs_6m` 仍是主要困难地图，`3s5z` 仍需要关注 seed 稳定性。现在应先完成统一配置在剩余五张地图上的验证，再决定是否进行新的机制修改。

## 2. AMCO 结构和研究定位

AMCO 是当前 PMIX-MLP 路线的实现。状态先经过无约束 encoder，随后进入只对个体 Q 单调、对状态自由的输入层：

$$
z = g_{\theta}(s),
\qquad
h_0 = \phi_q(\mathbf q) + \alpha\,\phi_z(z),
\qquad \alpha=0.3.
$$

其中 `q_weight` 通过正负部分分解保证每个 $q_i$ 的偏导非负，`state_weight` 不受单调约束。后续 AMCO monotone layers 对输入整体保持非递减，最终输出为：

$$
Q_{\mathrm{tot}} = M_{\theta}(\mathbf q,s) + V_{\psi}(s) + R(\mathbf q).
$$

当前标准配置令 $R(\mathbf q)=0$，因此实际训练使用：

$$
Q_{\mathrm{tot}} = M_{\theta}(\mathbf q,s) + V_{\psi}(s).
$$

`V(s)` 只依赖状态，不改变固定状态下的 joint-action 排序；$M$ 对每个个体 Q 单调，因此保留 IGM 所需的 argmax inclusion。代码仍保留可选 residual 和退火实现，便于复现实验记录，但 residual 不属于当前统一范式配置。

## 3. 当前代码和配置

当前 checkout：`master`，工作区干净。关键版本：

- 配置提交：`decbd4b`；
- mixer 提交：`a37ec41`；
- 当前实验脚本提交：`cb83ad1`。

关键文件：

- [AMCO 配置](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/src/config/algs/amco.yaml)
- [AMCO mixer](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/src/modules/mixers/amco_monotone.py)
- [诊断和 learner 接口](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/src/learners/q_learner.py)
- [剩余地图运行脚本](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/run_amco_activation_screen.sh)

当前主配置如下：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `mixer` | `amco` | 使用 AMCO partial-monotone mixer |
| `amco_mono_activation` | `centered_softplus` | 单调 Q 分支激活 |
| `amco_mono_softplus_beta` | `2.0` | centered Softplus 平滑度 |
| `amco_state_embed_dim` | `104` | 状态 encoder 输出维度 |
| `amco_mono_hidden_dim` | `96` | 单调 mixer 隐层宽度 |
| `amco_mono_depth` | `4` | 单调主干深度，满足当前 AMCO 结构约束 |
| `amco_state_encoder_depth` | `2` | 状态 encoder 深度 |
| `amco_state_activation` | `silu` | 状态 encoder 激活 |
| `amco_state_value_dim` | `32` | `V(s)` 隐层宽度 |
| `amco_state_value_activation` | `relu` | 状态值分支激活 |
| `amco_state_input_scale` | `0.3` | 状态输入项的统一缩放 |
| `mixer_diagnostics` | `True` | 开启诊断，不改变训练目标 |
| `credit_near_zero_threshold` | `1e-4` | credit 梯度近零阈值 |

训练脚本的共同设置：`t_max=2,050,000`，seeds 为 `1, 41, 141`，TensorBoard 开启，默认每 `10,000` 环境步记录一次 learner/mixer 统计。脚本按 12 个 GPU 先运行四张地图的 12 个任务，再运行 `bane_vs_bane` 的三个任务。

## 4. 实验演化

### 4.1 早期 residual 和 map-specific 调参

早期 modify 版本曾在 AMCO、HLL、MonoKAN 中加入 `sum(Q_i)` residual，并给 AMCO 使用地图相关的 state scale。历史记录显示：

- AMCO 的 residual annealing 在 MMM2 上有稳定作用；
- 固定 residual 容易把 mixer 推向 VDN-like shortcut；
- residual 对 AMCO、HLL、MonoKAN 的影响方向不一致，不能直接上升为统一 PMIX 机制；
- 因此当前标准范式移除 Q residual，也移除 AMCO 的 map-specific scale。

详细历史记录见 [2026-07-02 实验记录](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/实验记录/2026年07月02日/2026年07月02日.md)。

### 4.2 激活函数筛选

已比较的候选包括：ReLU、未中心化 Softplus、ELU、centered Softplus beta=1 和 centered Softplus beta=2。

- ELU 在 `5m_vs_6m` 上几乎失败；
- 未中心化 Softplus 在 `3s5z` 上出现明显 seed 分裂；
- centered Softplus 显著改善了训练稳定性；
- beta=1 和 beta=2 的差异具有地图依赖，不支持简单的训练过程 beta annealing；
- beta=2 在综合样本效率上最好，因此冻结为当前统一候选。

此前曲线分析得到的近似归一化 AUC 如下。该表用于激活函数筛选，不等价于冻结协议下的最终 PMIX 主结果；`3s5z` 的 beta=2 早期汇总只含两个完整 seed。

| 地图 | ReLU | centered beta=1 | centered beta=2 |
|---|---:|---:|---:|
| `3s5z` | 0.501 | 0.426 | 0.523* |
| `2c_vs_64zg` | 0.477 | 0.482 | **0.676** |
| `5m_vs_6m` | 0.319 | **0.429** | 0.406 |
| `3s_vs_5z` | 0.500 | 0.517 | **0.565** |

约 11 个完整配对 run 的总体汇总为：beta=2 平均 AUC 约 `0.544`，beta=1 约 `0.463`，ReLU 约 `0.455`。final-10 均值分别约为 `0.816`、`0.827` 和 `0.780`。这表示 beta=2 主要改善早期学习速度，最终胜率并不在所有地图都严格最高。

## 5. 当前已完成的 beta=2 结果

本地可见的 beta=2 日志位于：

- [beta=2 diagnostic screen](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/parallel_logs/amco_centered_softplus_b20_diagnostic_screen)
- [scale=0.3 follow-up](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/parallel_logs/amco_centered_softplus_b20_scale030_validation)
- [ReLU/早期候选 Sacred 结果](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/results_Paradigm_v2)
- [centered beta=1 Sacred 结果](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/results_Paradigm_v3)
- [已有结果曲线](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/result_plot/_figure)

下表为每个 seed 训练末期最后一次 `test_battle_won_mean`，不是单个 episode 的胜率。`5m_vs_6m` 三个 seed 和 `3s5z/seed41` 来自 scale=0.3 follow-up；其余已有三 seed 日志来自 beta=2 diagnostic screen，实际使用的地图 scale 也是 0.3。

| 地图 | seed=1 | seed=41 | seed=141 | 三 seed均值 | 主要观察 |
|---|---:|---:|---:|---:|---|
| `2c_vs_64zg` | 0.9250 | 0.9437 | 0.8812 | 0.9166 | 状态/Q 交互最强，但最终性能较好 |
| `3s5z` | 0.8938 | 0.7312 | 0.7750 | 0.8000 | seed 方差明显，早期 seed41 曾 OOM |
| `5m_vs_6m` | 0.7375 | 0.6937 | 0.6625 | 0.6979 | 当前四图中最困难，仍需优化稳定性 |
| `3s_vs_5z` | 0.9500 | 0.9125 | 0.9375 | 0.9333 | 统一 beta=2 后表现稳定 |

本地日志中 `3s5z/seed41` 的第一轮只运行到约 `10k` steps，随后因同时占用 GPU 导致 CUDA OOM；这不是算法失败。后续单独的 scale=0.3 重跑完整结束，末期测试胜率为 `0.7312`。

## 6. 诊断结果和机制解释

当前 learner/mixer 诊断包括：

- 个体 credit 梯度的 mean、std、median、p10、p90、CV、entropy、最大份额；
- `credit_grad_near_zero_frac` 和 `credit_grad_negative_frac`；
- AMCO 的 Q branch、state interaction branch、`V(s)` 和 mixing output RMS；
- `amco_state_to_q_ratio` 与 `amco_v_to_m_ratio`；
- 状态置换前后的 `amco_state_credit_delta` 和 `amco_state_credit_share_delta`；
- 各 monotone layer 的 preactivation、非正比例和 activation slope；
- Q 权重、state 权重和 state encoder 的梯度范数。

已有诊断支持以下判断：

1. **没有 credit 梯度死亡。** 已观察到 `credit_grad_negative_frac=0`、`credit_grad_near_zero_frac=0`，说明 AMCO 的非负单调梯度约束没有把个体 credit 压成零。
2. **没有明显的 centered Softplus 饱和。** 后期第一层 activation slope 的 p10 通常约为 `0.31--0.53`，仍有有效梯度。
3. **状态对 Q 的影响具有地图依赖。** 代表性后期 `state_to_q` 比例约为：`2c_vs_64zg=1.34`，`5m_vs_6m=0.65`，`3s5z` 约 `0.27--0.80`，`3s_vs_5z` 约 `0.14--0.22`。
4. **状态改变 credit 分配的程度也具有地图依赖。** 代表性 `state_credit_share_delta` 约为：`2c_vs_64zg=0.0236`，`5m_vs_6m=0.0273`，`3s5z=0.010`，`3s_vs_5z=0.00049`。因此不能要求所有地图具有相同的 state-sensitive credit。
5. **`5m_vs_6m` 的问题更像优化难度而非梯度死亡。** 该地图后期曾出现较大的 TD error 和 grad norm、较低的 activation slope p10，以及偏负的 preactivation；但近零梯度比例仍为零，暂时没有证据支持增加均匀 credit shortcut。

## 7. 待运行地图和服务器命令

当前标准九图中，已经完成 beta=2 验证的四张为：

- `3s5z`
- `2c_vs_64zg`
- `5m_vs_6m`
- `3s_vs_5z`

当前脚本准备运行的剩余五张为：

- `1c3s5z`
- `2s3z`
- `MMM2`
- `27m_vs_30m`
- `bane_vs_bane`

服务器运行命令：

```bash
conda activate pymarl
CUDA_DEVICES="0 1 2 3 4 5 6 7 8 9 10 11" \
  bash /Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/run_amco_activation_screen.sh
```

服务器目录通常为 `/data/workspace/wxr/pymarl-monotone-qmix`，运行脚本会在其中写入 `results/sacred/amco` 和 `results/tb_logs`。当前脚本不再传递 `amco_state_input_scale_by_map.*`，以避免 Sacred 报：

```text
ConfigAddedError: Added new config entry that is not used anywhere
```

## 8. 数据来源和可比性注意事项

- 当前本地 checkout 没有 `results/` 目录；最新 beta=2 的可见证据主要是 `parallel_logs`，而 [results_Paradigm_v2](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/results_Paradigm_v2) 保存 ReLU/早期 beta 候选，[results_Paradigm_v3](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/results_Paradigm_v3) 保存 centered beta=1 等结果。服务器上的 `results/` 应在下一轮实验完成后同步回来。
- 旧日志的 Sacred 配置仍可能包含历史 `amco_state_input_scale_by_map` 字典，不能据此判断当前配置仍启用 map-specific scale。以当前 [amco.yaml](/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/src/config/algs/amco.yaml) 和新服务器 run 的最终 Sacred config 为准。
- 不同版本的 best 曲线、不同 residual、不同 map scale 不能直接组成严格公平的平均排名。后续应以同一配置、同一 `t_max`、同一三个 seed 和同一测试窗口重新汇总 AUC、final-10、seed std。
- 当前研究决策优先看 credit 和 state/Q 诊断，不仅看最终 win rate。任何新机制都应先在 baseline-preserving 诊断上做一因素实验。

## 9. 后续对话的最短接替流程

1. 先检查服务器五张待跑地图是否全部产生三个 seed 的 Sacred/TensorBoard 结果。
2. 读取每个 run 的最终 config，确认 `centered_softplus`、beta=2、scale=0.3、Q residual=0，且没有旧的 map override。
3. 统一计算 final-10、AUC、seed 均值和标准差。
4. 对 `MMM2`、`27m_vs_30m` 和 `5m_vs_6m` 优先查看 `state_to_q`、state-credit delta、slope p10、TD error 和 grad norm。
5. 在统一九图结果完成前，不增加 VDN-like residual、credit floor 或额外 auxiliary loss；若需要修改，保持一次只改变一个因素。
