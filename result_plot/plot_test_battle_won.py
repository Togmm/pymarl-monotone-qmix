#!/usr/bin/env python3
"""Plot test_battle_won_mean curves across seeds.

Examples
--------
Read TensorBoard logs when tensorboard is installed:

    python result_plot/plot_test_battle_won.py \
        --results result_plot/mmm2 \
        --source tensorboard \
        --stat median \
        --smooth 0.95 \
        --out result_plot/_figure/bane_vs_bane_test_win.png

Fallback to Sacred cout.txt logs:

    python result_plot/plot_test_battle_won.py \
        --results result_plot/3s_vs_5z \
        --source sacred \
        --stat median \
        --out result_plot/_figure/mmm2_test_win.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


TAG = "test_battle_won_mean"


@dataclass
class Curve:
    method: str
    run: str
    map_name: str
    steps: np.ndarray
    values: np.ndarray


def normalize_method(name: str, collapse_variants: bool = False) -> str:
    """Normalize run names such as hll_seed41__2026... to hll.

    TensorBoard runs often look like:
        tb_logs/hll_nov_2026-06-15_22-00-14
        tb_logs/monokan_seed41__2026-06-17_09-46-35
        tb_logs/qmix__2026-06-13_14-51-50
    """

    name = name.replace("\\", "/").split("/")[-1]

    # Remove timestamp suffixes.
    name = re.sub(r"__\d{4}-\d{2}-\d{2}.*$", "", name)
    name = re.sub(r"_\d{4}-\d{2}-\d{2}.*$", "", name)

    # Remove explicit seed labels.
    name = re.sub(r"_seed\d+$", "", name)

    # Optional: merge hll_v and hll_nov into hll, etc.
    if collapse_variants:
        name = re.sub(r"_(nov|no_v|v)$", "", name)

    return name


def read_map_name_from_config(config_path: Path) -> str:
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        return "unknown"
    env_args = cfg.get("env_args") or {}
    return str(env_args.get("map_name") or "unknown")


def read_method_from_config(config_path: Path, collapse_variants: bool) -> str:
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        return config_path.parent.parent.name
    raw = str(cfg.get("name") or cfg.get("mixer") or config_path.parent.parent.name)
    return normalize_method(raw, collapse_variants=collapse_variants)


def read_sacred_curves(results: Path, collapse_variants: bool) -> List[Curve]:
    """Read test_battle_won_mean from Sacred cout.txt files."""

    curves: List[Curve] = []
    sacred = results / "sacred"
    if not sacred.exists():
        return curves

    stat_re = re.compile(r"Recent Stats \| t_env:\s*(\d+)")
    tag_re = re.compile(rf"{re.escape(TAG)}:\s*([-+0-9.eE]+)")

    for cout_path in sacred.glob("**/cout.txt"):
        run_dir = cout_path.parent
        if not run_dir.name.isdigit():
            continue

        config_path = run_dir / "config.json"
        method = read_method_from_config(config_path, collapse_variants)
        map_name = read_map_name_from_config(config_path)

        steps: List[int] = []
        values: List[float] = []
        current_step: Optional[int] = None

        for line in cout_path.read_text(errors="ignore").splitlines():
            step_match = stat_re.search(line)
            if step_match:
                current_step = int(step_match.group(1))

            tag_match = tag_re.search(line)
            if tag_match and current_step is not None:
                steps.append(current_step)
                values.append(float(tag_match.group(1)))

        if steps:
            curves.append(
                Curve(
                    method=method,
                    run=str(run_dir.relative_to(results)),
                    map_name=map_name,
                    steps=np.asarray(steps, dtype=np.float64),
                    values=np.asarray(values, dtype=np.float64),
                )
            )

    return curves


def try_read_tensorboard_curves(results: Path, collapse_variants: bool) -> List[Curve]:
    """Read TensorBoard event files.

    Requires tensorboard to be installed in the Python environment:
        pip install tensorboard
    """

    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore
            EventAccumulator,
        )
    except Exception as exc:
        raise RuntimeError(
            "TensorBoard is not installed. Install it or use --source sacred."
        ) from exc

    curves: List[Curve] = []
    tb_root = results / "tb_logs"
    if not tb_root.exists():
        return curves

    event_files = sorted(tb_root.glob("**/events.out.tfevents*"))
    for event_file in event_files:
        run_dir = event_file.parent
        run_name = str(run_dir.relative_to(results))
        method = normalize_method(run_name, collapse_variants=collapse_variants)

        acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        try:
            acc.Reload()
        except Exception:
            continue

        tags = acc.Tags().get("scalars", [])
        if TAG not in tags:
            continue

        events = acc.Scalars(TAG)
        if not events:
            continue

        steps = np.asarray([event.step for event in events], dtype=np.float64)
        values = np.asarray([event.value for event in events], dtype=np.float64)

        # TB log names do not always encode the map. Use a single panel unless
        # Sacred is used, or override the title with --title.
        curves.append(
            Curve(
                method=method,
                run=run_name,
                map_name="unknown",
                steps=steps,
                values=values,
            )
        )

    return curves


def filter_curves(
    curves: Iterable[Curve],
    methods: Optional[Sequence[str]],
    maps: Optional[Sequence[str]],
    min_final_step: float,
    min_points: int,
) -> List[Curve]:
    method_set = set(methods or [])
    map_set = set(maps or [])
    selected = []
    for curve in curves:
        if method_set and curve.method not in method_set:
            continue
        if map_set and curve.map_name not in map_set:
            continue
        if len(curve.steps) < min_points:
            continue
        if float(np.nanmax(curve.steps)) < min_final_step:
            continue
        order = np.argsort(curve.steps)
        selected.append(
            Curve(
                method=curve.method,
                run=curve.run,
                map_name=curve.map_name,
                steps=curve.steps[order],
                values=curve.values[order],
            )
        )
    return selected


def make_grid(curves: Sequence[Curve], step_max: Optional[float], points: int) -> np.ndarray:
    if step_max is None:
        step_max = max(float(np.nanmax(curve.steps)) for curve in curves)
    return np.linspace(0.0, step_max, points)


def interpolate_curve(curve: Curve, grid: np.ndarray) -> np.ndarray:
    """Interpolate one seed to the common grid.

    Before the first evaluation point, test win is treated as 0. After the last
    point, the final observed value is carried forward.
    """

    steps = curve.steps
    values = curve.values
    unique_steps, unique_indices = np.unique(steps, return_index=True)
    unique_values = values[unique_indices]
    return np.interp(grid, unique_steps, unique_values, left=0.0, right=unique_values[-1])


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    window = int(window)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(y, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def tensorboard_smooth(y: np.ndarray, weight: float) -> np.ndarray:
    """TensorBoard-style exponential smoothing.

    TensorBoard's UI smoothing slider is essentially an exponential moving
    average where larger values make the curve smoother. A value of 0 disables
    smoothing; 0.8 is a common presentation setting.
    """

    if weight <= 0:
        return y
    if weight >= 1:
        raise ValueError("--smooth must be less than 1.0")

    smoothed = np.empty_like(y, dtype=np.float64)
    last = float(y[0])
    smoothed[0] = last
    for i in range(1, len(y)):
        last = last * weight + (1.0 - weight) * float(y[i])
        smoothed[i] = last
    return smoothed


def apply_smoothing(y: np.ndarray, smooth_weight: float, smooth_window: int) -> np.ndarray:
    y = tensorboard_smooth(y, smooth_weight)
    return moving_average(y, smooth_window)


def aggregate(
    curves: Sequence[Curve],
    grid: np.ndarray,
    stat: str,
    band: str,
    smooth_weight: float,
    smooth_window: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    by_method: Dict[str, List[np.ndarray]] = defaultdict(list)
    for curve in curves:
        by_method[curve.method].append(interpolate_curve(curve, grid))

    output: Dict[str, Dict[str, np.ndarray]] = {}
    for method, arrs in by_method.items():
        mat = np.vstack(arrs)
        if stat == "mean":
            center = np.nanmean(mat, axis=0)
        elif stat == "median":
            center = np.nanmedian(mat, axis=0)
        else:
            raise ValueError(f"Unknown stat: {stat}")

        if band == "iqr":
            low = np.nanpercentile(mat, 25, axis=0)
            high = np.nanpercentile(mat, 75, axis=0)
        elif band == "minmax":
            low = np.nanmin(mat, axis=0)
            high = np.nanmax(mat, axis=0)
        elif band == "std":
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            low = mean - std
            high = mean + std
        elif band == "sem":
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0) / math.sqrt(max(1, mat.shape[0]))
            low = mean - sem
            high = mean + sem
        else:
            raise ValueError(f"Unknown band: {band}")

        output[method] = {
            "center": apply_smoothing(center, smooth_weight, smooth_window),
            "low": apply_smoothing(low, smooth_weight, smooth_window),
            "high": apply_smoothing(high, smooth_weight, smooth_window),
            "n": np.asarray([mat.shape[0]], dtype=np.int64),
        }

    return output


def method_sort_key(method: str) -> Tuple[int, str]:
    preferred = [
        "qmix",
        "hll",
        "hll_nov",
        "hll_v",
        "monokan",
        "monokan_nov",
        "smm",
        "amco",
        "smnn",
        "lmn",
    ]
    try:
        return (preferred.index(method), method)
    except ValueError:
        return (len(preferred), method)


def plot_panel(
    ax: plt.Axes,
    grid: np.ndarray,
    aggregated: Dict[str, Dict[str, np.ndarray]],
    title: str,
    ylabel: str,
) -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, method in enumerate(sorted(aggregated, key=method_sort_key)):
        data = aggregated[method]
        x = grid / 1_000_000.0
        center = data["center"] * 100.0
        low = np.clip(data["low"] * 100.0, 0.0, 100.0)
        high = np.clip(data["high"] * 100.0, 0.0, 100.0)
        color = colors[idx % len(colors)]
        label = f"{method} (n={int(data['n'][0])})"
        ax.plot(x, center, label=label, color=color, linewidth=2.4)
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("T (mil)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(-5, 105)
    ax.set_xlim(grid[0] / 1_000_000.0, grid[-1] / 1_000_000.0)
    ax.grid(True, color="#c9c9c9", linewidth=1.0, alpha=0.85)
    ax.legend(fontsize=9, frameon=False, loc="best")


def print_summary(curves: Sequence[Curve]) -> None:
    grouped: Dict[str, List[Curve]] = defaultdict(list)
    for curve in curves:
        grouped[curve.method].append(curve)

    print("Selected runs:")
    for method in sorted(grouped, key=method_sort_key):
        finals = [float(curve.values[-1]) for curve in grouped[method]]
        peaks = [float(np.nanmax(curve.values)) for curve in grouped[method]]
        final_txt = ", ".join(f"{v:.4f}" for v in finals)
        peak_txt = ", ".join(f"{v:.4f}" for v in peaks)
        print(
            f"  {method:14s} n={len(grouped[method])} "
            f"final=[{final_txt}] peak=[{peak_txt}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("/Users/wxr/FIle/RL/MARL/Code/pymarl/pymarl/results"),
        help="Path to PyMARL results directory.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "tensorboard", "sacred"],
        default="auto",
        help="Where to read curves from.",
    )
    parser.add_argument("--tag", default=TAG, help="Only test_battle_won_mean is supported by default.")
    parser.add_argument("--out", type=Path, default=Path("test_battle_won_mean.png"))
    parser.add_argument("--title", default=None)
    parser.add_argument("--stat", choices=["median", "mean"], default="median")
    parser.add_argument("--band", choices=["iqr", "minmax", "std", "sem"], default="iqr")
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="TensorBoard-style exponential smoothing weight, e.g. 0.8.",
    )
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--step-max", type=float, default=2_000_000.0)
    parser.add_argument("--min-final-step", type=float, default=1_900_000.0)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--maps", nargs="*", default=None)
    parser.add_argument(
        "--collapse-variants",
        action="store_true",
        help="Merge names like hll_v/hll_nov into hll.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(8.0, 5.0),
        metavar=("W", "H"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tag != TAG:
        raise ValueError("This script is intentionally focused on test_battle_won_mean.")

    curves: List[Curve] = []
    if args.source in ("auto", "tensorboard"):
        try:
            curves = try_read_tensorboard_curves(args.results, args.collapse_variants)
        except RuntimeError as exc:
            if args.source == "tensorboard":
                raise
            print(f"[warn] {exc}")

    if not curves and args.source in ("auto", "sacred"):
        curves = read_sacred_curves(args.results, args.collapse_variants)

    curves = filter_curves(
        curves,
        methods=args.methods,
        maps=args.maps,
        min_final_step=args.min_final_step,
        min_points=args.min_points,
    )
    if not curves:
        raise SystemExit("No curves found after filtering.")

    print_summary(curves)

    # If Sacred provides multiple maps, draw one panel per map. TensorBoard logs
    # usually do not encode map names, so they produce a single unknown panel.
    by_map: Dict[str, List[Curve]] = defaultdict(list)
    for curve in curves:
        by_map[curve.map_name].append(curve)

    n_maps = len(by_map)
    fig, axes = plt.subplots(
        n_maps,
        1,
        figsize=(args.figsize[0], args.figsize[1] * n_maps),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, (map_name, map_curves) in zip(axes[:, 0], sorted(by_map.items())):
        grid = make_grid(map_curves, args.step_max, args.points)
        aggregated = aggregate(
            map_curves,
            grid=grid,
            stat=args.stat,
            band=args.band,
            smooth_weight=args.smooth,
            smooth_window=args.smooth_window,
        )
        title = args.title or (map_name if map_name != "unknown" else TAG)
        ylabel = f"{args.stat.title()} Test Win (%)"
        plot_panel(ax, grid, aggregated, title=title, ylabel=ylabel)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()