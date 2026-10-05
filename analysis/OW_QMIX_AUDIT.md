# OW-QMIX baseline audit

审计日期：2026-10-04

参考实现：作者仓库 [oxwhirl/wqmix](https://github.com/oxwhirl/wqmix)，其中
`hysteretic_qmix=True` 就是 OW-QMIX；论文为
[Weighted QMIX](https://arxiv.org/abs/2006.10800)。

## 结论

`src/learners/wqmix_learner.py` 的算法主干是正确的，可以作为项目内的 OW-QMIX
baseline：

1. 局部 agent utility 经过标准 QMIX mixer 得到 `Q_tot`。
2. 独立的 central MAC 和非单调 feed-forward mixer 表示无限制的
   `Q_hat*`，不与局部 QMIX 网络共享参数。
3. 下一状态动作由局部 QMIX 网络的 greedy action（`double_q=True` 时为在线
   网络）选择，再由 central target 网络计算 TD target。
4. OW 权重等价于论文公式 (5) 和作者代码：
   `w=1` 当 `Q_tot < y`，否则 `w=alpha`。当前代码通过
   `td_error < 0` 实现，完全等价。
5. CW 分支的 `u == argmax Q_tot` 或 `y > Q_hat*(s, argmax Q_tot)` 判定也与
   作者 `max_q_learner.py` 一致。

代码已通过 `python3 -m compileall -q src` 语法检查。

## 与作者实现的有意差异

- 本项目把本地 mixer 通过 `args.mixer` 注册表选择，默认仍是标准 `qmix`；
  central mixer 固定为 `QMixerCentralFF`。这不改变 OW-QMIX 的目标或权重定义。
- 本项目使用 `central_rnn_hidden_dim` 配置 central agent 隐层维度；作者仓库
  复用 `rnn_hidden_dim`。当前值都是 64，因此网络结构等价。
- `target_local_mixer` 被维护和保存，但 Weighted QMIX 的 bootstrap 使用的是
  `target_central_mixer`，所以它不参与 target 计算；这是冗余状态，不是算法错误。

## 公平比较协议

当前 `ow_qmix.yaml` 与 `qmix.yaml` 对齐了影响优化预算的共同参数：

- `epsilon_start/finish=1.0/0.05`、`epsilon_anneal_time=50000`
- `batch_size=32`、`buffer_size=5000`
- `lr=5e-4`、RMSProp `alpha=0.99`、`eps=1e-5`
- `gamma=0.99`、`grad_norm_clip=10`、`target_update_interval=200`
- 相同的 map、seed、`t_max` 和评估间隔

OW-QMIX 额外的 central `Q_hat*` 网络是算法本身的必要组成部分，不能为了参数量
一致而删除；比较时应报告这是额外训练开销，并确保所有算法使用相同环境交互步数、
seed 集合和评估方法。`w=0.1` 只属于 OW 的算法超参数，不应改动 QMIX baseline
来“匹配”它。

## 复现边界

作者 README 和论文脚注明确指出，论文 SMAC 结果使用
`SC2.4.6.2.69232`，不同 StarCraft 版本的结果不可直接比较。项目现有日志记录为
`Version: B75689 (SC2.4.10)`。因此：

- 现有结果可以作为“同一项目、同一 SC2.4.10 环境下的公平 OW-QMIX 对比”；
- 不能作为论文曲线或论文数值的严格复现；
- 若需要论文级复现，应另外安装并固定 `SC2.4.6.2.69232`，并在所有 baseline
  使用同一版本后重新运行。

## 建议

保留当前实现和公平配置；在实验表格/README 中明确写出环境版本、`w=0.1`、
50k exploration schedule，以及“项目内公平对比而非论文数值复现”的定位。
