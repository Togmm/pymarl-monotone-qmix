# ICES baselines

This repository now exposes four ICES compositions:

* `ices`: ICES-QMIX;
* `ices_pmix_mlp`: ICES with the existing AMCO/PMIX MLP mixer;
* `ices_pmix_lattice`: ICES with the HLL lattice mixer;
* `ices_pmix_kan`: ICES with the MonoKAN mixer.

The exploitation branch uses the configured observation-based recurrent agent
and QMIX/PMIX mixer.  During centralized training, a global
Gaussian transition predictor and a leave-one-agent-out predictor estimate a
bounded diagonal-Gaussian KL surprise for every agent.  A second recurrent
policy receives the global state, is trained with an intrinsic policy-gradient
loss and value baseline, and is mixed into behaviour action selection with a
decaying ICES ratio.  The privileged policy and state are never used by the
evaluation branch.

For `ices_pmix_*`, the exploitation branch reuses the existing
`kaleidoscope_rnn_1r3`, mask diversity regularization, and all
mixer-specific AMCO/HLL/MonoKAN settings.  The ICES intrinsic branch is the
only added training mechanism.

The ICES paper implementation uses Pyro conditional VAEs and SVI.  This port
uses an equivalent observable mechanism with diagonal Gaussian predictors and
Adam NLL updates so it remains dependency-free and can compose with all local
mixers.  The TD target remains the framework's existing double-Q target with
availability, termination and filled masks.

## Minimal runs

```bash
# EPyMARL / LBF CPU smoke run
PYTHONPATH=src python src/main.py --config=ices --env-config=lbf with \
  env_args.key=lbforaging:Foraging-8x8-2p-2f-coop-v3 \
  env_args.time_limit=3 t_max=3 batch_size_run=1 buffer_size=1 batch_size=1 \
  use_cuda=False

# EPyMARL uses the existing SMACv2 and LBF launchers
DRY_RUN=True ALGS="ices ices_pmix_mlp ices_pmix_lattice ices_pmix_kan" \
  SCENARIOS="protoss_5_vs_5" SEEDS="1" \
  bash run_parallel_epymarl_smacv2.sh
DRY_RUN=True ALGS="ices ices_pmix_mlp ices_pmix_lattice ices_pmix_kan" \
  TASKS="8x8_2p_2f_coop" SEEDS="1" \
  bash run_parallel_epymarl_lbf.sh
sbatch submit_epymarl_smacv2_ices.slurm
sbatch submit_epymarl_smacv2_ices_pmix.slurm
sbatch submit_epymarl_lbf_ices.slurm
sbatch submit_epymarl_lbf_ices_pmix.slurm

# PyMARL uses SMAC v1
DRY_RUN=True ALGS="ices ices_pmix_mlp ices_pmix_lattice ices_pmix_kan" \
  bash run_parallel_pymarl_smac.sh
sbatch submit_pymarl_smac_ices.slurm
sbatch submit_pymarl_smac_ices_pmix.slurm
```

All four configurations keep the existing comparison budget (batch, replay,
learning rate, epsilon schedule and target interval).  The additional ICES
parameters are `ices_int_ratio=0.1`, `ices_int_finish=0.1`, `ices_int_lr=0.01`,
`ices_int_critic_lr=0.001`, `ices_int_entropy_coef=0.1`, and
`ices_world_lr=1e-4`.
