#!/usr/bin/env python3
"""Plot test_return_mean curves across seeds.

Examples
--------
Read TensorBoard logs when tensorboard is installed:

    python3 result_plot/plot_test_battle_won_paper.py \
        --results result_plot/smacv1/3s_vs_5z \
                  result_plot/smacv1/5m_vs_6m \
                  result_plot/smacv1/bane_vs_bane \
                  result_plot/smacv1/mmm2 \
        --source tensorboard \
        --stat median \
        --smooth 0.98 \
        --steps 2000000 \
        --auto-y \
        --tag test_battle_won_mean \
        --ylabel "Median Test Win (%)" \
        --out result_plot/_figure

The paper-style defaults are median aggregation, EMA smoothing with
``--smooth 0.98``, adaptive y-limits, and a figure-level legend.  The output
file is named ``<environment>-paper_figure.png`` inside ``--out``.

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


TAG = "test_return_mean"
# TAG = "test_battle_won_mean"

# Colour-blind-friendly Okabe-Ito inspired palette.  The blue/green choices
# differ in both hue and brightness so red-green weakness is less problematic.
METHOD_COLORS = {
    "qmix": "#0072b2",     # blue
    "hll": "#e69f00",      # orange
    "monokan": "#009e73",  # bluish green
    "amco": "#56b4e9",     # sky blue
    "ices": "#cc79a7",     # reddish purple
    "ices_pmix_kan": "#f0e442",  # yellow
    "kaleidoscope_qmix": "#d55e00",     # vermillion
    "kaleidoscope_pmix_kan": "#000000", # black
    "cw_qmix": "#8c8c8c",  # gray
    "ow_qmix": "#332288",  # indigo
    "qplex": "#aa4499",    # purple
    "s2q": "#117733",      # dark green
}

# Fallback colours are kept separate from the named-method colours above so an
# unlisted method cannot silently reuse a colour already used in the legend.
FALLBACK_COLORS = (
    "#88ccaa", "#ddcc77", "#cc6677", "#44aa99", "#999933",
    "#6699cc", "#661100", "#882255", "#774411", "#dd9977",
)

# Publication-facing names.  Unknown methods keep their original name.
METHOD_LABELS = {
    "qmix": "QMIX",
    "amco": "PMIX-MLP",
    "hll": "PMIX-Lattice",
    "monokan": "PMIX-KAN",
    "ices": "ICES",
    "ices_pmix_kan": "ICES-PMIX-KAN",
    "kaleidoscope_qmix": "Kaleidoscope-QMIX",
    "kaleidoscope_pmix_kan": "Kaleidoscope-PMIX-KAN",
}


def method_color(method: str, fallback_index: int, colors: Sequence[str]) -> str:
    """Return a stable colour, including for named method variants."""

    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    # Keep hll_v/hll_nov (and similar variants) in the base method's colour.
    base_method = method.split("_", 1)[0]
    return METHOD_COLORS.get(
        base_method, FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]
    )


def method_label(method: str) -> str:
    """Return the publication label while preserving unknown method names."""

    return METHOD_LABELS.get(method, method)


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
    return str(
        env_args.get("map_name")
        or env_args.get("key")
        or env_args.get("env_name")
        or cfg.get("env")
        or "unknown"
    )


def read_method_from_config(config_path: Path, collapse_variants: bool) -> str:
    try:
        cfg = json.loads(config_path.read_text())
    except Exception:
        return config_path.parent.parent.name
    raw = str(cfg.get("name") or cfg.get("mixer") or config_path.parent.parent.name)
    return normalize_method(raw, collapse_variants=collapse_variants)


def read_sacred_curves(
    results: Path, collapse_variants: bool, tag: str
) -> List[Curve]:
    """Read a scalar tag from Sacred cout.txt files.

    Both Sacred's normal layout and the flattened layout used by the archived
    plotting inputs are supported::

        sacred/<method>/<run_id>/cout.txt
        sacred/<method>/cout.txt
    """

    curves: List[Curve] = []
    sacred = results / "sacred"
    if not sacred.exists():
        return curves

    stat_re = re.compile(r"Recent Stats \| t_env:\s*(\d+)")
    tag_re = re.compile(rf"{re.escape(tag)}:\s*([-+0-9.eE]+)")

    # Read only the selected results directory's expected Sacred layout.
    cout_paths = list(sacred.glob("*/cout.txt"))
    for method_dir in sacred.iterdir():
        if not method_dir.is_dir():
            continue
        cout_paths.extend(method_dir.glob("*/cout.txt"))

    for cout_path in sorted(cout_paths):
        run_dir = cout_path.parent
        is_numbered_run = run_dir.name.isdigit()
        is_flattened_run = run_dir.parent == sacred and (
            run_dir / "config.json"
        ).is_file()
        if not (is_numbered_run or is_flattened_run):
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


def try_read_tensorboard_curves(
    results: Path, collapse_variants: bool, tag: str
) -> List[Curve]:
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

    # Do not recursively walk arbitrary descendants: runs and event files are
    # expected at the two levels immediately below this selected directory.
    event_files = list(tb_root.glob("events.out.tfevents*"))
    for run_dir in tb_root.iterdir():
        if run_dir.is_dir():
            event_files.extend(run_dir.glob("events.out.tfevents*"))
    event_files = sorted(event_files)
    for event_file in event_files:
        run_dir = event_file.parent
        run_name = str(run_dir.relative_to(results))
        method, map_name = read_tensorboard_identity(
            run_dir.name, collapse_variants, results.name
        )

        acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        try:
            acc.Reload()
        except Exception:
            continue

        tags = acc.Tags().get("scalars", [])
        if tag not in tags:
            continue

        events = acc.Scalars(tag)
        if not events:
            continue

        steps = np.asarray([event.step for event in events], dtype=np.float64)
        values = np.asarray([event.value for event in events], dtype=np.float64)

        # EPyMARL LBF run names encode the Gymnasium key; other TB logs may not.
        curves.append(
            Curve(
                method=method,
                run=run_name,
                map_name=map_name,
                steps=steps,
                values=values,
            )
        )

    return curves


def read_tensorboard_identity(
    run_name: str, collapse_variants: bool, map_hint: Optional[str] = None
) -> Tuple[str, str]:
    """Extract method and Gymnasium LBF map from common EPyMARL run names.

    Examples include ``qmix_lbforaging:Foraging-8x8-..._seed1_...`` and
    ``qmix_seed1_lbforaging:Foraging-8x8-..._...``.  Other environments use
    the method prefix in ``<method>_<map>_seed...`` and are assigned an
    unknown map.
    """

    lbf_match = re.match(
        r"^(?P<method>.+?)(?:_seed\d+)?_lbforaging:"
        r"(?P<map>Foraging-.+?)(?:_seed\d+)?_\d{4}-",
        run_name,
    )
    if lbf_match:
        return (
            normalize_method(lbf_match.group("method"), collapse_variants),
            f"lbforaging:{lbf_match.group('map')}",
        )

    # When one map directory is supplied, its name cleanly separates method
    # names containing underscores (for example cw_qmix) from the map name.
    # Current LBF archives omit the ``Foraging-`` prefix in run names and use
    # a hyphen before the map, for example
    # ``cw_qmix-10x10-3p-3f-v3_seed41_2026-...``.  A few older runs put the
    # seed between the method and map instead:
    # ``monokan_seed141-2s-8x8-2p-2f-coop-v3_2026-...``.
    if map_hint:
        map_variants = [map_hint]
        if map_hint.startswith("Foraging-"):
            map_variants.append(map_hint.removeprefix("Foraging-"))

        for map_name in map_variants:
            escaped_map = re.escape(map_name)
            patterns = (
                rf"^(?P<method>.+?)[_-]{escaped_map}_seed\d+_+\d{{4}}-",
                rf"^(?P<method>.+?)_seed\d+-{escaped_map}_\d{{4}}-",
            )
            for pattern in patterns:
                suffix_match = re.match(pattern, run_name)
                if suffix_match:
                    return (
                        normalize_method(
                            suffix_match.group("method"), collapse_variants
                        ),
                        map_hint,
                    )

    # Keep methods with underscores in their names separate when run names
    # use a double underscore before the timestamp.  The generic fallback
    # below only retains the first word of a method name.
    for method in (
        "ices_pmix_kan",
        "kaleidoscope_pmix_kan",
        "kaleidoscope_qmix",
    ):
        if run_name.startswith(f"{method}_"):
            return method, "unknown"

    # Non-LBF EPyMARL runs conventionally use
    # ``<method>_<map>_seed<seed>_<timestamp>``.  Previously the complete
    # run name (minus the timestamp/seed) was returned here, e.g.
    # ``qmix_terran_5_vs_5``.  That made the method order, and consequently
    # its colour, depend on the environment.  Keep the method prefix only;
    # an optional ``_nov``/``_v`` suffix is retained for variant experiments.
    method_match = re.match(
        r"^(?P<method>[^_]+(?:_(?:nov|no_v|v))?)(?:_|$)", run_name
    )
    method = method_match.group("method") if method_match else run_name
    return normalize_method(method, collapse_variants), "unknown"


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

    Before the first evaluation point, test return is treated as 0. After the
    last point, the final observed value is carried forward.
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
    if smooth_weight > 0 and smooth_window > 1:
        raise ValueError(
            "Choose either --smooth (EMA) or --smooth-window (moving average), "
            "not both."
        )
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
        "ices",
        "ices_pmix_kan",
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
    auto_y: bool = False,
    legend_counts: bool = False,
) -> List[plt.Line2D]:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    y_values = []
    for idx, method in enumerate(sorted(aggregated, key=method_sort_key)):
        data = aggregated[method]
        x = grid / 1_000_000.0
        center = data["center"] * 100.0
        low = np.clip(data["low"] * 100.0, 0.0, 100.0)
        high = np.clip(data["high"] * 100.0, 0.0, 100.0)
        y_values.extend((center, low, high))
        # Explicit colours keep method identity stable across panels. Unknown
        # methods use the high-contrast fallback palette above.
        color = method_color(method, idx, colors)
        label = method_label(method)
        if legend_counts:
            label = f"{label} (n={int(data['n'][0])})"
        ax.plot(x, center, label=label, color=color, linewidth=2.4,
                solid_capstyle="round")
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
    ax.set_xlabel("Environment steps (million)", fontsize=13, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=7)
    if auto_y and y_values:
        finite_values = np.concatenate(y_values)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size:
            y_min = float(np.min(finite_values))
            y_max = float(np.max(finite_values))
            span = max(y_max - y_min, 1.0)
            padding = 0.05 * span
            ax.set_ylim(max(0.0, y_min - padding), min(100.0, y_max + padding))
        else:
            ax.set_ylim(-5, 105)
    else:
        ax.set_ylim(-5, 105)
    ax.set_xlim(grid[0] / 1_000_000.0, grid[-1] / 1_000_000.0)
    ax.grid(True, color="#bdbdbd", linewidth=0.8, alpha=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # The figure-level legend is assembled by plot_results.  Returning the
    # handles keeps the panel renderer useful without duplicating legends.
    return ax.get_legend_handles_labels()[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="One or more PyMARL results directories; all are combined into one figure.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "tensorboard", "sacred"],
        default="auto",
        help="Where to read curves from.",
    )
    parser.add_argument(
        "--tag",
        default=TAG,
        help="Scalar tag to plot (default: test_return_mean).",
    )
    parser.add_argument(
        "--ylabel",
        default=None,
        help="Y-axis label (default: derived from --stat and --tag).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("result_plot/_figure"),
        help="Output directory; saved as <environment>-paper_figure.png.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--stat", choices=["median", "mean"], default="median")
    parser.add_argument("--band", choices=["iqr", "minmax", "std", "sem"], default="iqr")
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.98,
        help="TensorBoard-style exponential smoothing weight (default: 0.98).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered moving-average width in interpolated plotting points.",
    )
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument(
        "--steps",
        "--step-max",
        dest="steps",
        type=float,
        default=2_000_000.0,
        help="Total training steps and plot x-axis upper bound (default: 2000000).",
    )
    parser.add_argument(
        "--min-final-step",
        type=float,
        default=None,
        help="Minimum final step for a run (default: 95%% of --steps).",
    )
    parser.add_argument(
        "--auto-y",
        action="store_true",
        default=True,
        help="Adapt each panel's y-axis to its plotted values (default).",
    )
    parser.add_argument(
        "--fixed-y",
        action="store_true",
        help="Disable adaptive y-limits and use 0--100 for every panel.",
    )
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
        default=(5.4, 4.2),
        metavar=("W", "H"),
        help="Width and height of each subplot in inches.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=None,
        help="Number of subplot columns (default: 2 for <=4 panels, otherwise 3).",
    )
    parser.add_argument(
        "--legend-counts",
        action="store_true",
        help="Append the number of runs to method names in the global legend.",
    )
    return parser.parse_args()


def output_path(output_dir: Path, results: Path) -> Path:
    """Build ``<environment>_<map>.png`` from a results directory path."""

    normalized = results.resolve()
    return output_dir / f"{normalized.parent.name}_{normalized.name}.png"


def _read_and_group_results(
    args: argparse.Namespace, results: Path
) -> List[Tuple[Path, str, List[Curve]]]:
    """Read one results directory and return exactly one plot panel.

    Each ``--results`` path already identifies a single environment/map.  Old
    TensorBoard run names do not always encode that map, so grouping again by
    ``Curve.map_name`` can incorrectly split one directory into both its real
    map and an ``unknown`` panel.
    """

    curves: List[Curve] = []
    if args.source in ("auto", "tensorboard"):
        try:
            curves = try_read_tensorboard_curves(
                results, args.collapse_variants, args.tag
            )
        except RuntimeError as exc:
            if args.source == "tensorboard":
                raise
            print(f"[warn] {exc}")

    if not curves and args.source in ("auto", "sacred"):
        curves = read_sacred_curves(results, args.collapse_variants, args.tag)

    curves = filter_curves(
        curves,
        methods=args.methods,
        maps=args.maps,
        min_final_step=args.min_final_step,
        min_points=args.min_points,
    )
    if not curves:
        raise RuntimeError(f"No curves found after filtering: {results}")

    return [(results, results.name, curves)]


def plot_results(args: argparse.Namespace) -> Path:
    """Plot every ``--results`` directory on one publication-style canvas."""

    panels: List[Tuple[Path, str, List[Curve]]] = []
    failures: List[str] = []
    print(f"Loading {len(args.results)} result directories for one combined figure...")
    for results in args.results:
        try:
            panels.extend(_read_and_group_results(args, results))
        except (RuntimeError, ValueError) as exc:
            failures.append(str(exc))
            print(f"[error] {exc}")
    if not panels:
        raise RuntimeError("No plottable result directories were found.")
    if failures:
        print(f"[warn] Skipped {len(failures)} result director(y/ies).")

    total_runs = sum(len(map_curves) for _, _, map_curves in panels)
    print(
        f"Plotting one combined figure with {len(panels)} panels "
        f"from {total_runs} selected runs."
    )

    n_panels = len(panels)
    ncols = args.ncols or (2 if n_panels <= 4 else 3)
    if ncols < 1:
        raise ValueError("--ncols must be positive.")
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(args.figsize[0] * ncols, args.figsize[1] * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axes_flat = list(axes.flat)
    handles: Dict[str, plt.Line2D] = {}
    labels: Dict[str, str] = {}

    for ax, (results, map_name, map_curves) in zip(axes_flat, panels):
        grid = make_grid(map_curves, args.steps, args.points)
        aggregated = aggregate(
            map_curves,
            grid=grid,
            stat=args.stat,
            band=args.band,
            smooth_weight=args.smooth,
            smooth_window=args.smooth_window,
        )
        title = args.title or (map_name if map_name != "unknown" else results.name)
        if args.ylabel:
            ylabel = args.ylabel
        elif "battle_won" in args.tag:
            ylabel = f"{args.stat.title()} Test Win (%)"
        else:
            ylabel = f"{args.stat.title()} Normalized Episode Return (%)"
        plot_panel(
            ax,
            grid,
            aggregated,
            title=title,
            ylabel=ylabel,
            auto_y=(args.auto_y and not args.fixed_y),
            legend_counts=args.legend_counts,
        )
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            labels.setdefault(label, label)
            handles.setdefault(label, handle)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Hide unused cells while retaining a clean rectangular grid.
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    display_order = {"QMIX": 0, "PMIX-MLP": 1, "PMIX-Lattice": 2, "PMIX-KAN": 3}
    ordered = sorted(
        labels,
        key=lambda value: (display_order.get(value, 100), value.lower()),
    )
    if ordered:
        fig.legend(
            [handles[label] for label in ordered],
            [labels[label] for label in ordered],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(len(ordered), 4),
            frameon=False,
            fontsize=14,
            handlelength=2.5,
            columnspacing=2.0,
        )

    # Reserve a little extra space for the top legend and keep subplot spacing
    # close to the compact layout used in the reference paper figure.
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.08, top=0.86,
        wspace=0.28, hspace=0.38,
    )
    environment_names = sorted({results.parent.name for results, _, _ in panels})
    environment = environment_names[0] if len(environment_names) == 1 else "combined"
    destination = args.out / f"{environment}-paper_figure.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved combined figure to {destination}")
    return destination


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be greater than 0.")
    if args.min_final_step is None:
        args.min_final_step = 0.9 * args.steps
    if args.min_final_step < 0:
        raise ValueError("--min-final-step must not be negative.")

    try:
        plot_results(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
