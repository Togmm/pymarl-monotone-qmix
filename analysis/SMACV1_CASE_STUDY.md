# PyMARL SMACv1 case-study evaluator

This directory contains the SMACv1-only evaluator:
`smacv1_case_study_eval.py`. It imports only this repository's `src`
packages and the installed SMACv1 dependency. It does not import EPyMARL.

The evaluator loads the PyMARL Sacred JSON and `agent.th`, runs greedy
episodes for QMIX/MonoKAN, saves native frames, observations, states, action
targets and unit snapshots, then writes `events.csv`, `seed_summary.csv`,
`bootstrap_stats.csv` and `selection.json`.

SMACv1 seeds are constructor seeds. The script creates a fresh native SMAC
environment for every episode so all methods consume the same
`episode_seeds.json` values.

## Run

```bash
cd /labmount/users/202535331/MARL/pymarl-monotone-qmix
export SDL_VIDEODRIVER=dummy

python3 analysis/smacv1_case_study_eval.py collect \
  --output analysis_data/smacv1/MMM2 \
  --map MMM2 --episodes 100 --overwrite \
  --model qmix,1,/labmount/users/202535331/MARL/pymarl-monotone-qmix/case_study/qmix_MMM2_seed41__2026-10-04_00-33-25/2000069 \
  --model monokan,1,/labmount/users/202535331/MARL/pymarl-monotone-qmix/case_study/monokan_MMM2_seed41__2026-10-04_00-33-25/2000057 \
  --config-json qmix=/labmount/users/202535331/MARL/pymarl-monotone-qmix/results/sacred/qmix/2/config.json \
  --config-json monokan=/labmount/users/202535331/MARL/pymarl-monotone-qmix/results/sacred/monokan/1/config.json

python3 analysis/smacv1_case_study_eval.py analyse \
  --data analysis_data/smacv1/5m_vs_6m \
  --selection-mode complex

python3 analysis/smacv1_case_study_eval.py plot \
  --data analysis_data/smacv1/5m_vs_6m
```

For a smoke test, use `--episodes 1 --no-frames` before the complete run.

## Textured native SC2 frames

The default `frame_*.png` files come from SMAC's lightweight pygame renderer.
It draws the height map and colored unit circles, which is why it looks flat.
To request the textured terrain and unit sprites produced by the StarCraft II
client, add `--native-rgb` to **collect**:

```bash
python3 analysis/smacv1_case_study_eval.py collect \
  --output analysis_data/smacv1/5m_vs_6m_native \
  --map 5m_vs_6m --episodes 1 --overwrite --native-rgb \
  --model qmix,1,/path/to/qmix_checkpoint \
  --model monokan,1,/path/to/monokan_checkpoint \
  --config-json qmix=/path/to/qmix_config.json \
  --config-json monokan=/path/to/monokan_config.json
```

This uses the SC2 `render_data.map` buffer and therefore requires a working
EGL/OSMesa software or hardware renderer.  If the installed SC2 binary cannot
produce RGB data, the collector falls back to the ordinary SMAC frame and the
manifest records `frame_source: smac_pygame_renderer`.  Native rendering is an
analysis-only option; it does not change the policy input, actions, or model
configuration.

## How the case and figure are selected

`analyse --selection-mode complex` pairs the same episode seed for both
methods and first keeps one-sided focus-fire outcomes.  It then chooses the
pair with the richest engagement using a deterministic score based on the
shorter episode length, target switches, attack-active steps, distinct targets,
and the number of agents that joined the first attack.  The selected episode,
score and every component are recorded in `selection.json`.  Use
`--selection-mode representative` to reproduce the earlier median-time-to-kill
selection rule.

The top row of the generated figure is QMIX and the bottom row is MonoKAN.  The
first panel is the native global SMAC view.  The other panels are enlarged
engagement crops with deterministic overlays: `A0`/`E0` unit IDs, health and
shield bars, a yellow ring around the focus target, yellow focus-fire arrows,
and cyan arrows for other attacks.  The lower panels retain the action timeline
and aggregate completion/time-to-kill statistics.  These additions are made
only while plotting saved trajectories; the simulator and training configs are
unchanged.
