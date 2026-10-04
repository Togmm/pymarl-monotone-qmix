#!/usr/bin/env python3
"""Standalone PyMARL evaluator for the SMACv1 case study.

This file intentionally lives in the PyMARL checkout and imports only that
checkout's ``src`` packages.  It loads the PyMARL ``agent.th`` checkpoint,
collects paired greedy episodes, detects focus-fire events, and writes the
case-study statistics/figure.  It never edits a training config.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "smacv1-case-study-v2"
DEFAULT_EPISODE_SEED_BASE = 2718281
ATTACK_OFFSET = 6


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    tmp.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, value)
    tmp.replace(path)


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def checkpoint_dir(path: Path) -> Path:
    if (path / "agent.th").is_file():
        return path
    leaves = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
    if not leaves:
        raise FileNotFoundError(f"No agent.th or numbered checkpoint under {path}")
    return max(leaves, key=lambda p: int(p.name))


def episode_seeds(count: int, base: int) -> List[int]:
    return [int(base + i) for i in range(int(count))]


def save_frame(path_without_suffix: Path, frame: Any) -> Optional[str]:
    if frame is None:
        return None
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    try:
        from PIL import Image

        filename = path_without_suffix.with_suffix(".png")
        filename.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image[..., :3], mode="RGB").save(filename)
        return filename.name
    except ImportError:
        filename = path_without_suffix.with_suffix(".npy")
        atomic_npy(filename, image)
        return filename.name


def render_rgb(env: Any) -> Optional[np.ndarray]:
    """Read a true SC2 RGB observation when native rendering is enabled.

    SMAC's usual ``env.render()`` is a lightweight pygame drawing made from
    the height map and unit circles.  Native mode asks the SC2 process for
    ``observation.render_data.map``; that buffer contains the actual textured
    StarCraft II terrain and unit sprites.  The lightweight renderer remains
    the fallback for machines without an EGL/OSMesa renderer.
    """
    observation = getattr(getattr(env, "_obs", None), "observation", None)
    render_data = getattr(observation, "render_data", None)
    if render_data is not None:
        try:
            from pysc2.lib import features
            if render_data.HasField("map"):
                return np.asarray(features.Feature.unpack_rgb_image(render_data.map))
        except (AttributeError, AssertionError, ImportError, ValueError):
            pass
    for call in (
        lambda: env.render(mode="rgb_array"),
        lambda: env.render(),
    ):
        try:
            frame = call()
        except (TypeError, RuntimeError, OSError):
            continue
        if frame is None:
            continue
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            return frame[..., :3]
    return None


class NativeRGBStarCraft2Env:
    """Factory wrapper that enables SC2's textured render buffer.

    SMACv1 hard-codes ``want_rgb=False`` and does not expose a render option in
    its constructor.  This subclass temporarily patches those two internal
    launch arguments while calling SMAC's own launch routine.  It lives only
    in this analysis script; no PyMARL source or Sacred configuration is
    changed.
    """

    def __new__(cls, **kwargs: Any) -> Any:
        from smac.env import StarCraft2Env
        # Build a per-instance subclass so the installed SMAC package remains
        # untouched for ordinary training/evaluation commands.
        class _NativeRGBEnv(StarCraft2Env):
            def _launch(self_inner):
                from smac.env.starcraft2 import starcraft2 as smac_impl

                original_get = smac_impl.run_configs.get
                original_interface = smac_impl.sc_pb.InterfaceOptions

                def patched_get(*args: Any, **kw: Any) -> Any:
                    run_config = original_get(*args, **kw)
                    original_start = run_config.start

                    def start_with_rgb(*start_args: Any, **start_kw: Any) -> Any:
                        start_kw["want_rgb"] = True
                        return original_start(*start_args, **start_kw)

                    run_config.start = start_with_rgb
                    return run_config

                def interface_with_rgb(*interface_args: Any, **interface_kw: Any) -> Any:
                    options = original_interface(*interface_args, **interface_kw)
                    # The map buffer is independent of the training
                    # observation/state dimensions.  Keep it moderate so a
                    # long collection does not exhaust host memory.
                    options.render.resolution.x = int(min(max(self_inner.window_size[0], 640), 1280))
                    options.render.resolution.y = int(min(max(self_inner.window_size[1], 640), 720))
                    options.render.minimap_resolution.x = 256
                    options.render.minimap_resolution.y = 256
                    return options

                smac_impl.run_configs.get = patched_get
                smac_impl.sc_pb.InterfaceOptions = interface_with_rgb
                try:
                    return super(_NativeRGBEnv, self_inner)._launch()
                finally:
                    smac_impl.run_configs.get = original_get
                    smac_impl.sc_pb.InterfaceOptions = original_interface

        return _NativeRGBEnv(**kwargs)


def install_native_rgb_factory() -> None:
    ensure_pymarl_imports()
    from envs import REGISTRY as env_registry
    env_registry["sc2"] = lambda **kwargs: NativeRGBStarCraft2Env(**kwargs)


class NullLogger:
    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


def _src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def ensure_pymarl_imports() -> None:
    src = str(_src_dir())
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)


@dataclass
class Policy:
    method: str
    training_seed: int
    checkpoint: Path
    config_path: Path
    config: Dict[str, Any]
    runner: Any
    mac: Any
    env_info: Dict[str, Any]

    @property
    def env(self) -> Any:
        return self.runner.env

    def close(self) -> None:
        try:
            self.runner.close_env()
        except Exception:
            try:
                self.env.close()
            except Exception:
                pass


def build_policy(method: str, training_seed: int, checkpoint: Path, config_path: Path,
                 map_name: str, use_cuda: bool, native_rgb: bool = False) -> Policy:
    """Build PyMARL's own BasicMAC from the saved agent checkpoint."""
    ensure_pymarl_imports()
    import torch as th
    from components.episode_buffer import EpisodeBatch
    from components.transforms import OneHot
    from controllers import REGISTRY as mac_registry
    from runners import REGISTRY as runner_registry
    if native_rgb:
        install_native_rgb_factory()

    config = dict(read_json(config_path))
    config["env"] = "sc2"
    config["batch_size_run"] = 1
    config["runner"] = "episode"
    config["use_cuda"] = bool(use_cuda)
    config["test_greedy"] = True
    config["render"] = False
    config.setdefault("runner_log_interval", 10**9)
    config.setdefault("epsilon_start", 1.0)
    config.setdefault("epsilon_finish", 0.05)
    config.setdefault("epsilon_anneal_time", 1)
    env_args = dict(config.get("env_args") or {})
    env_args["map_name"] = map_name
    env_args["seed"] = int(training_seed)
    config["env_args"] = env_args
    config["seed"] = int(training_seed)

    random.seed(training_seed)
    np.random.seed(training_seed)
    th.manual_seed(training_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(training_seed)
    args = SimpleNamespace(**config)
    args.device = "cuda" if use_cuda and th.cuda.is_available() else "cpu"
    runner = runner_registry["episode"](args, NullLogger())
    info = runner.get_env_info()
    args.n_agents = int(info["n_agents"])
    args.n_actions = int(info["n_actions"])
    args.state_shape = info["state_shape"]
    scheme = {
        "state": {"vshape": info["state_shape"]},
        "obs": {"vshape": info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (info["n_actions"],), "group": "agents", "dtype": th.int},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "reward": {"vshape": (1,)},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    # EpisodeRunner.new_batch is the authoritative PyMARL EpisodeBatch
    # constructor.  This temporary batch supplies its scheme to BasicMAC.
    probe = EpisodeBatch(scheme, groups, 1, int(info["episode_limit"]) + 1,
                         preprocess=preprocess, device=args.device)
    mac = mac_registry[args.mac](probe.scheme, groups, args)
    runner.setup(scheme, groups, preprocess, mac)
    leaf = checkpoint_dir(checkpoint)
    mac.load_models(str(leaf))
    if args.device == "cuda" and hasattr(mac, "cuda"):
        mac.cuda()
    if hasattr(mac, "agent"):
        mac.agent.eval()
    return Policy(method, training_seed, leaf, config_path, config, runner, mac, info)


def reset_episode(policy: Policy, seed: int) -> None:
    """Recreate native SMAC so the constructor seed is really applied."""
    ensure_pymarl_imports()
    from envs import REGISTRY as env_registry

    old = policy.runner.env
    env_args = dict(policy.config["env_args"])
    env_args["seed"] = int(seed)
    new_env = env_registry["sc2"](**env_args)
    try:
        old.close()
    except Exception:
        pass
    policy.runner.env = new_env
    policy.runner.episode_limit = new_env.episode_limit
    policy.env_info = new_env.get_env_info()
    new_env.reset()


def unit_value(unit: Any, name: str, default: Any = None) -> Any:
    value = getattr(unit, name, default)
    return default if value is None else value


def unit_position(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if hasattr(value, "x") and hasattr(value, "y"):
        return [float(value.x), float(value.y)]
    try:
        values = list(value)
        return [float(values[0]), float(values[1])]
    except (TypeError, IndexError, ValueError):
        return None


def snapshot(env: Any) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {"allies": [], "enemies": []}
    for side, attr in (("allies", "agents"), ("enemies", "enemies")):
        units = getattr(env, attr, {}) or {}
        items = units.items() if isinstance(units, Mapping) else enumerate(units)
        for index, unit in items:
            health = float(unit_value(unit, "health", 0.0) or 0.0)
            result[side].append({
                "index": int(index),
                "tag": int(unit_value(unit, "tag", index) or index),
                "unit_type": int(unit_value(unit, "unit_type", -1) or -1),
                "position": unit_position(unit_value(unit, "pos")),
                "health": health,
                "health_max": float(unit_value(unit, "health_max", health) or health),
                "shield": float(unit_value(unit, "shield", 0.0) or 0.0),
                "shield_max": float(unit_value(unit, "shield_max", 0.0) or 0.0),
                "alive": health > 0,
            })
    return result


def action_records(env: Any, before: Mapping[str, Any], actions: Sequence[int]) -> List[Dict[str, Any]]:
    allies = {x["index"]: x for x in before.get("allies", [])}
    enemies = {x["index"]: x for x in before.get("enemies", [])}
    medivac_id = getattr(env, "medivac_id", None)
    records = []
    for agent_id, raw in enumerate(actions):
        action = int(raw)
        actor = allies.get(agent_id, {})
        target_index = action - ATTACK_OFFSET if action >= ATTACK_OFFSET else None
        kind = "noop" if action == 0 else "stop" if action == 1 else "move" if 2 <= action <= 5 else "other"
        target = None
        if target_index is not None:
            is_medivac = medivac_id is not None and actor.get("unit_type") == int(medivac_id)
            target = (allies if is_medivac else enemies).get(target_index)
            kind = "heal" if is_medivac else "attack"
        records.append({
            "agent_id": agent_id, "action": action, "kind": kind,
            "target_index": target_index,
            "target_tag": None if target is None else target.get("tag"),
            "target_type": None if target is None else target.get("unit_type"),
            "actor_tag": actor.get("tag"), "actor_alive": actor.get("alive", False),
            "target_alive_before": None if target is None else target.get("alive", False),
        })
    return records


def collect_episode(policy: Policy, root: Path, episode_id: int, seed: int,
                    save_frames: bool, overwrite: bool) -> Dict[str, Any]:
    directory = root / safe_name(policy.method) / f"seed{policy.training_seed}" / f"episode_{episode_id:04d}"
    trajectory = directory / "trajectory.json"
    if trajectory.exists() and not overwrite:
        existing = read_json(trajectory)
        if int(existing.get("episode_seed", seed)) != int(seed):
            raise RuntimeError(f"{trajectory} has a different episode_seed; use --overwrite")
        return existing
    if directory.exists() and overwrite:
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reset_episode(policy, seed)
    env = policy.env
    batch = policy.runner.new_batch()
    policy.mac.init_hidden(batch_size=1)
    steps, states, observations = [], [], []
    total_reward = 0.0
    limit = int(policy.env_info["episode_limit"])
    for t in range(limit):
        state = np.asarray(env.get_state(), dtype=np.float32)
        obs = np.asarray(env.get_obs(), dtype=np.float32)
        avail = np.asarray(env.get_avail_actions(), dtype=np.int8)
        states.append(state.copy()); observations.append(obs.copy())
        before = snapshot(env)
        frame_name = None
        if save_frames:
            saved = save_frame(directory / "frames" / f"frame_{t:04d}", render_rgb(env))
            frame_name = None if saved is None else str(Path("frames") / saved)
        batch.update({"state": [state], "avail_actions": [avail], "obs": [obs]}, ts=t)
        import torch as th
        with th.no_grad():
            chosen = policy.mac.select_actions(batch, t_ep=t, t_env=0, test_mode=True)
        actions = chosen.detach().cpu().numpy().reshape(-1).astype(int).tolist()
        records = action_records(env, before, actions)
        reward, terminated, info = env.step(actions)
        info = dict(info or {})
        total_reward += float(reward)
        after = snapshot(env)
        batch.update({"actions": [actions], "reward": [(float(reward),)],
                      "terminated": [(bool(terminated and not info.get("episode_limit", False)),)]}, ts=t)
        steps.append({"t": t, "frame": frame_name, "state_index": t,
                      "obs_shape": list(obs.shape), "avail_actions": avail.tolist(),
                      "actions": actions, "action_records": records,
                      "entities_before": before, "entities_after": after,
                      "reward": float(reward), "terminated": bool(terminated),
                      "truncated": False, "info": info})
        if terminated:
            break
    atomic_npy(directory / "states.npy", np.asarray(states, dtype=np.float32))
    atomic_npy(directory / "observations.npy", np.asarray(observations, dtype=np.float32))
    record = {"schema_version": SCHEMA, "suite": "smacv1", "method": policy.method,
              "training_seed": policy.training_seed, "episode_id": episode_id,
              "episode_seed": seed, "checkpoint": str(policy.checkpoint),
              "config_json": str(policy.config_path), "episode_limit": limit,
              "length": len(steps), "total_reward": total_reward, "steps": steps}
    write_json(trajectory, record)
    return record


def parse_model(value: str) -> Tuple[str, int, Path, Optional[Path]]:
    fields = value.split(",", 3)
    if len(fields) < 3:
        raise ValueError("--model must be method,seed,checkpoint[,config.json]")
    return fields[0], int(fields[1]), Path(fields[2]).expanduser(), Path(fields[3]).expanduser() if len(fields) == 4 else None


def config_defaults(values: Iterable[str]) -> Dict[str, Path]:
    result = {}
    for value in values:
        method, path = value.split("=", 1)
        result[method] = Path(path).expanduser()
    return result


def collect(args: argparse.Namespace) -> None:
    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    defaults = config_defaults(args.config_json)
    specs = [parse_model(x) for x in args.model]
    if not specs:
        raise ValueError("At least one --model is required")
    seeds = episode_seeds(args.episodes, args.episode_seed_base)
    write_json(root / "episode_seeds.json", {"schema_version": SCHEMA, "base": args.episode_seed_base, "seeds": seeds})
    manifest = {"schema_version": SCHEMA, "suite": "smacv1", "map": args.map,
                "episodes_per_model": args.episodes,
                "frame_source": "sc2_native_rgb" if args.native_rgb else "smac_pygame_renderer",
                "models": []}
    for method, train_seed, checkpoint, inline in specs:
        config = inline or defaults.get(method)
        if config is None:
            raise ValueError(f"No PyMARL config for {method}; pass --config-json {method}=...")
        policy = build_policy(method, train_seed, checkpoint, config, args.map, args.use_cuda,
                              args.native_rgb)
        try:
            for episode_id, seed in enumerate(seeds):
                print(f"[smacv1/PyMARL] {method} seed={train_seed} episode={episode_id + 1}/{len(seeds)}", flush=True)
                collect_episode(policy, root, episode_id, seed, not args.no_frames, args.overwrite)
        finally:
            policy.close()
        snapshot_path = root / safe_name(method) / f"seed{train_seed}" / "config_snapshot.json"
        write_json(snapshot_path, policy.config)
        manifest["models"].append({"method": method, "training_seed": train_seed,
                                    "checkpoint": str(policy.checkpoint), "config_json": str(config),
                                    "framework": "pymarl", "config_snapshot": str(snapshot_path.relative_to(root)),
                                    "env_info": policy.env_info})
    write_json(root / "manifest.json", manifest)


def iter_records(root: Path) -> Iterator[Dict[str, Any]]:
    for path in sorted(root.glob("*/seed*/episode_*/trajectory.json")):
        yield read_json(path)


def index_side(snapshot_data: Mapping[str, Any], side: str) -> Dict[Any, Dict[str, Any]]:
    out = {}
    for item in snapshot_data.get(side, []):
        out[item.get("tag", item.get("index"))] = item
        out[item.get("index")] = item
    return out


def detect_event(record: Mapping[str, Any], horizon: int) -> Dict[str, Any]:
    steps = record.get("steps", [])
    onset = target = None
    for t, step in enumerate(steps):
        attacks = [x for x in step.get("action_records", []) if x.get("kind") == "attack" and x.get("actor_alive") and x.get("target_alive_before")]
        counts = Counter(x.get("target_tag") for x in attacks)
        shared = [k for k, n in counts.items() if n >= 2]
        if shared:
            onset, target = t, shared[0]
            break
    event = {"schema_version": SCHEMA, "suite": "smacv1", "method": record.get("method"),
             "training_seed": record.get("training_seed"), "episode_id": record.get("episode_id"),
             "episode_seed": record.get("episode_seed"), "episode_length": len(steps),
             "engagement_detected": onset is not None, "onset": onset, "target_tag": target,
             "time_to_kill": None, "target_switches": 0, "ally_deaths": 0,
             "medivac_heal": False, "focus_fire_success": False,
             # These fields make the case-study selection reproducible.  They
             # describe how much coordination is visible around the first
             # engagement, rather than selecting only by the outcome.
             "shared_attackers": 0, "active_attack_steps": 0,
             "unique_targets": 0, "attack_actions": 0}
    if onset is None:
        return event
    end = min(len(steps) - 1, onset + horizon)
    first_attacks = [x for x in steps[onset].get("action_records", [])
                     if x.get("kind") == "attack" and x.get("actor_alive")
                     and x.get("target_alive_before") and x.get("target_tag") == target]
    event["shared_attackers"] = len({x.get("agent_id") for x in first_attacks})
    all_targets = set()
    for t in range(onset, end + 1):
        step_attacks = [x for x in steps[t].get("action_records", [])
                        if x.get("kind") == "attack" and x.get("actor_alive")
                        and x.get("target_alive_before")]
        if step_attacks:
            event["active_attack_steps"] += 1
            event["attack_actions"] += len(step_attacks)
            all_targets.update(x.get("target_tag") for x in step_attacks
                               if x.get("target_tag") is not None)
        enemy = index_side(steps[t].get("entities_after", {}), "enemies").get(target)
        if enemy is not None and not enemy.get("alive", False):
            event["time_to_kill"] = t - onset + 1
            break
    for t in range(onset, end + 1):
        before_units = steps[t].get("entities_before", {}).get("allies", [])
        after = index_side(steps[t].get("entities_after", {}), "allies")
        event["ally_deaths"] += sum(
            1 for unit in before_units
            if unit.get("alive") and not after.get(unit.get("tag", unit.get("index")), {}).get("alive", False)
        )
        if any(x.get("kind") == "heal" for x in steps[t].get("action_records", [])):
            event["medivac_heal"] = True
    previous = {}
    for t in range(onset, end + 1):
        for action in steps[t].get("action_records", []):
            if action.get("kind") != "attack":
                continue
            agent, selected = action.get("agent_id"), action.get("target_tag")
            if agent in previous and previous[agent] != selected:
                event["target_switches"] += 1
            previous[agent] = selected
    event["unique_targets"] = len(all_targets)
    event["focus_fire_success"] = event["time_to_kill"] is not None and event["ally_deaths"] <= 1
    return event


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def bootstrap(values: Sequence[float], seed: int) -> Tuple[float, float]:
    vals = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if vals.size == 0: return math.nan, math.nan
    if vals.size == 1: return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(10000, vals.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyse(args: argparse.Namespace) -> None:
    root = Path(args.data).expanduser().resolve()
    events = [detect_event(record, args.horizon) for record in iter_records(root)]
    write_csv(root / "events.csv", events)
    grouped = defaultdict(list)
    for event in events:
        grouped[(event["method"], event["training_seed"])].append(event)
    summaries = []
    for (method, seed), group in sorted(grouped.items(), key=lambda x: (str(x[0][0]), int(x[0][1]))):
        detected = [x for x in group if x["engagement_detected"]]
        success = [x for x in group if x["focus_fire_success"]]
        ttks = [x["time_to_kill"] for x in detected if x["time_to_kill"] is not None]
        summaries.append({"method": method, "training_seed": seed, "episodes": len(group),
                          "engagement_rate": len(detected) / len(group) if group else math.nan,
                          "focus_fire_completion_rate": len(success) / len(group) if group else math.nan,
                          "median_time_to_kill": float(np.median(ttks)) if ttks else math.nan,
                          "median_target_switches": float(np.median([x["target_switches"] for x in detected])) if detected else math.nan})
    write_csv(root / "seed_summary.csv", summaries)
    stats = []
    for method in sorted({x["method"] for x in summaries}):
        values = [float(x["focus_fire_completion_rate"]) for x in summaries if x["method"] == method]
        low, high = bootstrap(values, int(hashlib.sha256(str(method).encode()).hexdigest()[:8], 16))
        stats.append({"method": method, "metric": "focus_fire_completion_rate", "mean": float(np.nanmean(values)) if values else math.nan, "ci_low": low, "ci_high": high, "n_seeds": len(values)})
    write_csv(root / "bootstrap_stats.csv", stats)
    selection = select_pair(events, mode=args.selection_mode)
    write_json(root / "selection.json", selection)


def pair_complexity(left: Mapping[str, Any], right: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Score the amount of visible coordination in a paired episode.

    The score is deliberately computed from both methods before the winner is
    chosen.  Longer engagements, more target changes, more attack actions and
    more distinct targets produce a richer replay.  This avoids repeatedly
    selecting a very short opening exchange that says little about policy
    differences.
    """
    components = {
        "shared_attackers": int(left.get("shared_attackers", 0)) + int(right.get("shared_attackers", 0)),
        "episode_length": min(int(left.get("episode_length", 0)), int(right.get("episode_length", 0))),
        "target_switches": int(left.get("target_switches", 0)) + int(right.get("target_switches", 0)),
        "active_attack_steps": int(left.get("active_attack_steps", 0)) + int(right.get("active_attack_steps", 0)),
        "unique_targets": int(left.get("unique_targets", 0)) + int(right.get("unique_targets", 0)),
        "ally_deaths": int(left.get("ally_deaths", 0)) + int(right.get("ally_deaths", 0)),
    }
    # The weights keep episode duration and target switching as the dominant
    # terms while giving multi-agent attack density a visible contribution.
    score = (1.0 * components["episode_length"]
             + 0.75 * components["target_switches"]
             + 0.35 * components["active_attack_steps"]
             + 0.20 * components["unique_targets"]
             + 1.50 * components["shared_attackers"]
             + 0.50 * components["ally_deaths"])
    return float(score), components


def select_pair(events: Sequence[Mapping[str, Any]], mode: str = "complex") -> Dict[str, Any]:
    methods = [x for x in ("qmix", "monokan") if any(e["method"] == x for e in events)]
    methods += sorted({str(e["method"]) for e in events}.difference(methods))
    if len(methods) < 2: return {"schema_version": SCHEMA, "status": "insufficient_methods", "methods": methods}
    left, right = methods[:2]
    by_key = {(e["method"], e["training_seed"], e["episode_id"]): e for e in events}
    successful = [e for e in events if e["focus_fire_success"] and e["time_to_kill"] is not None]
    median = float(np.median([e["time_to_kill"] for e in successful])) if successful else None
    candidates = []
    for seed in sorted({e["training_seed"] for e in events}):
        for episode in sorted({e["episode_id"] for e in events if e["training_seed"] == seed}):
            a, b = by_key.get((left, seed, episode)), by_key.get((right, seed, episode))
            if a is None or b is None: continue
            if bool(a["focus_fire_success"]) ^ bool(b["focus_fire_success"]):
                winner = a if a["focus_fire_success"] else b
                score, components = pair_complexity(a, b)
                ttk_distance = (abs(winner["time_to_kill"] - median)
                                if median is not None and winner["time_to_kill"] is not None else math.inf)
                candidates.append({"score": score, "components": components,
                                   "ttk_distance": ttk_distance, "seed": int(seed),
                                   "episode": int(episode), "winner": winner["method"]})
    requested_mode = mode
    mode = "complex_one_sided_success" if mode == "complex" else "one_sided_success"
    if not candidates:
        mode = "median_ttk_difference_fallback"
        fallback = []
        for seed in sorted({e["training_seed"] for e in events}):
            for episode in sorted({e["episode_id"] for e in events if e["training_seed"] == seed}):
                a, b = by_key.get((left, seed, episode)), by_key.get((right, seed, episode))
                if a and b and a["time_to_kill"] is not None and b["time_to_kill"] is not None:
                    fallback.append((abs(a["time_to_kill"] - b["time_to_kill"]), int(seed), int(episode), None))
        if not fallback: return {"schema_version": SCHEMA, "status": "no_eligible_pair", "methods": methods}
        target = float(np.median([x[0] for x in fallback]))
        candidates = [{"score": -abs(x[0] - target), "components": {},
                       "ttk_distance": abs(x[0] - target), "seed": x[1],
                       "episode": x[2], "winner": x[3]} for x in fallback]
    if mode == "complex_one_sided_success":
        # Highest complexity wins; robust time-to-kill is only a tie breaker.
        chosen = min(candidates, key=lambda x: (-x["score"], x["ttk_distance"], x["seed"], x["episode"]))
    else:
        chosen = min(candidates, key=lambda x: (x["ttk_distance"], x["seed"], x["episode"]))
    seed, episode, winner = chosen["seed"], chosen["episode"], chosen["winner"]
    return {"schema_version": SCHEMA, "status": "selected", "pairing": "same_initial_episode_seed",
            "selection_mode": mode, "requested_selection_mode": requested_mode,
            "candidate_count": len(candidates), "complexity_score": chosen["score"],
            "complexity_components": chosen["components"], "methods": methods,
            "training_seed": seed, "episode_id": episode,
            "episode_seed": next(e["episode_seed"] for e in events if e["method"] == left and e["training_seed"] == seed and e["episode_id"] == episode),
            "successful_method": winner, "left": by_key[(left, seed, episode)], "right": by_key[(right, seed, episode)]}


def load_frame(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    if path.suffix == ".npy":
        try:
            return np.asarray(np.load(path))[..., :3]
        except OSError:
            return None
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))
    except (ImportError, OSError):
        return None


def _pixel_position(position: Any, width: int, height: int) -> Optional[Tuple[float, float]]:
    """Project native SMAC world coordinates onto its square renderer.

    SMACv1's renderer uses a 32 x 32 world for the maps used here.  The
    projection is therefore the same one used by the native viewer, so the
    labels and arrows land on top of the actual unit sprites.
    """
    try:
        x, y = float(position[0]), float(position[1])
    except (TypeError, IndexError, ValueError):
        return None
    scale = min(float(width), float(height)) / 32.0
    return x * scale, y * scale


def _draw_arrow(draw: Any, start: Tuple[float, float], end: Tuple[float, float],
                color: Tuple[int, int, int], width: int = 2) -> None:
    draw.line([start, end], fill=color, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = end
    left = (tip[0] - 10 * ux + 4 * px, tip[1] - 10 * uy + 4 * py)
    right = (tip[0] - 10 * ux - 4 * px, tip[1] - 10 * uy - 4 * py)
    draw.polygon([tip, left, right], fill=color)


def annotate_smac_frame(frame: np.ndarray, step: Mapping[str, Any], focus_tag: Any,
                        zoom: bool = False) -> np.ndarray:
    """Add unit IDs, health/shield bars, focus rings and attack arrows.

    Native SMAC frames intentionally use tiny ``Ma`` sprites.  The overlay is
    generated from the saved simulator snapshot, so it remains deterministic
    and does not alter the environment or any training configuration.
    """
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, int(width / 58)))
        small_font = ImageFont.truetype("DejaVuSans.ttf", max(10, int(width / 78)))
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    before = step.get("entities_before", {})
    units: Dict[Tuple[str, Any], Mapping[str, Any]] = {}
    positions: Dict[Tuple[str, Any], Tuple[float, float]] = {}
    for side, prefix, color in (("allies", "A", (40, 220, 70)), ("enemies", "E", (235, 75, 85))):
        for unit in before.get(side, []) or []:
            key = (side, unit.get("tag", unit.get("index")))
            pos = _pixel_position(unit.get("position"), width, height)
            if pos is None:
                continue
            units[key] = unit; positions[key] = pos

    # Draw commands first, then sprites and labels remain legible above them.
    for action in step.get("action_records", []) or []:
        if action.get("kind") != "attack":
            continue
        actor = units.get(("allies", action.get("actor_tag")))
        target_key = ("enemies", action.get("target_tag"))
        target = units.get(target_key)
        if actor is None or target is None:
            continue
        start, end = positions[("allies", action.get("actor_tag"))], positions[target_key]
        is_focus = action.get("target_tag") == focus_tag
        _draw_arrow(draw, start, end, (255, 218, 55) if is_focus else (60, 190, 220), 4 if is_focus else 2)

    focus_pos = None
    for (side, tag), unit in units.items():
        pos = positions[(side, tag)]
        alive = bool(unit.get("alive", float(unit.get("health", 0.0) or 0.0) > 0))
        color = ((40, 220, 70) if side == "allies" else (235, 75, 85)) if alive else (100, 100, 100)
        radius = max(11, int(width / 42))
        if tag == focus_tag and side == "enemies":
            focus_pos = pos
            draw.ellipse((pos[0] - radius - 7, pos[1] - radius - 7,
                          pos[0] + radius + 7, pos[1] + radius + 7),
                         outline=(255, 220, 45), width=4)
        # A dark outline is visible on both the olive map and the brown border.
        draw.ellipse((pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius),
                     fill=color, outline=(15, 15, 15), width=2)
        if not alive:
            draw.line((pos[0] - radius + 3, pos[1] - radius + 3,
                       pos[0] + radius - 3, pos[1] + radius - 3), fill=(255, 255, 255), width=2)
            draw.line((pos[0] + radius - 3, pos[1] - radius + 3,
                       pos[0] - radius + 3, pos[1] + radius - 3), fill=(255, 255, 255), width=2)
        index = unit.get("index", "?")
        label = ("A" if side == "allies" else "E") + str(index)
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text((pos[0] - (bbox[2] - bbox[0]) / 2, pos[1] - (bbox[3] - bbox[1]) / 2 - 1),
                  label, fill=(255, 255, 255), font=small_font)
        max_health = max(float(unit.get("health_max", 0.0) or 0.0), 1.0)
        health = max(0.0, float(unit.get("health", 0.0) or 0.0))
        bar_w, bar_h = max(28, int(width / 22)), max(4, int(width / 180))
        left, top = pos[0] - bar_w / 2, pos[1] - radius - bar_h - 5
        draw.rectangle((left, top, left + bar_w, top + bar_h), fill=(35, 35, 35), outline=(0, 0, 0))
        draw.rectangle((left, top, left + bar_w * min(1.0, health / max_health), top + bar_h), fill=(70, 235, 90))
        shield_max = float(unit.get("shield_max", 0.0) or 0.0)
        if shield_max > 0:
            shield = max(0.0, float(unit.get("shield", 0.0) or 0.0))
            st = top - bar_h - 2
            draw.rectangle((left, st, left + bar_w, st + bar_h), fill=(35, 35, 35), outline=(0, 0, 0))
            draw.rectangle((left, st, left + bar_w * min(1.0, shield / shield_max), st + bar_h), fill=(70, 170, 255))

    # Compact legend and focus readout make the plot self-contained when it is
    # copied into a paper or a slide without the surrounding CSV files.
    legend = "A: ally   E: enemy   bars: HP / shield   arrows: attack"
    draw.rounded_rectangle((8, 8, min(width - 8, 500), 34), radius=5, fill=(0, 0, 0), outline=(220, 220, 220))
    draw.text((15, 13), legend, fill=(255, 255, 255), font=small_font)
    if focus_pos is not None:
        focus_unit = next((u for (s, t), u in units.items() if s == "enemies" and t == focus_tag), {})
        suffix = " (down)" if not focus_unit.get("alive", True) else ""
        text = f"FOCUS E{focus_unit.get('index', '?')}{suffix}"
        draw.rounded_rectangle((8, 40, 150, 68), radius=5, fill=(70, 55, 0), outline=(255, 220, 45))
        draw.text((15, 45), text, fill=(255, 235, 100), font=small_font)

    if not zoom:
        return np.asarray(image)

    # Crop a square around the engagement and upscale it so IDs and health
    # bars can be read in the 2 x 4 comparison panel.
    if not positions:
        return np.asarray(image)
    xs = [p[0] for p in positions.values()]; ys = [p[1] for p in positions.values()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) + 150
    span = min(float(width), max(260.0, span))
    left = int(round(cx - span / 2)); top = int(round(cy - span / 2))
    left = max(0, min(width - int(span), left)); top = max(0, min(height - int(span), top))
    crop = image.crop((left, top, left + int(span), top + int(span)))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return np.asarray(crop.resize((width, height), resampling))


def plot(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(args.data).expanduser().resolve()
    selection = read_json(root / "selection.json")
    if selection.get("status") != "selected":
        raise RuntimeError("selection.json has no eligible paired episode")
    methods = selection["methods"]
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(4, 4, height_ratios=(2.0, 2.0, 1.1, 1.0))
    axes = [[fig.add_subplot(grid[row, col]) for col in range(4)] for row in range(2)]
    offsets = (-4, -1, 0, 4)
    for row, method in enumerate(methods):
        record_path = root / safe_name(method) / f"seed{selection['training_seed']}" / f"episode_{int(selection['episode_id']):04d}" / "trajectory.json"
        record = read_json(record_path); event = detect_event(record, args.horizon); onset = event.get("onset") or 0
        episode_root = record_path.parent
        for col, offset in enumerate(offsets):
            t = max(0, min(len(record["steps"]) - 1, onset + offset)); step = record["steps"][t]
            frame = None
            if step.get("frame"):
                frame = load_frame(episode_root / step["frame"])
            if frame is not None:
                # Keep one global view for spatial context and use enlarged
                # engagement crops for the remaining columns.
                frame = annotate_smac_frame(frame, step, event.get("target_tag"), zoom=col > 0)
            axes[row][col].imshow(frame if frame is not None else np.zeros((100, 160, 3), dtype=np.uint8))
            axes[row][col].axis("off"); axes[row][col].set_title(f"{method} onset{offset:+d}")
        axes[row][0].set_title(
            f"{method}  |  onset={event.get('onset')}  shared={event.get('shared_attackers', 0)}\n"
            f"TTK={event.get('time_to_kill', '—')}  switches={event.get('target_switches', 0)}  deaths={event.get('ally_deaths', 0)}",
            fontsize=9)
    timeline = fig.add_subplot(grid[2, :2])
    stats_ax = fig.add_subplot(grid[2, 2:])
    for lane, method in enumerate(methods):
        record_path = root / safe_name(method) / f"seed{selection['training_seed']}" / f"episode_{int(selection['episode_id']):04d}" / "trajectory.json"
        record = read_json(record_path); event = detect_event(record, args.horizon); onset = event.get("onset") or 0
        for step in record.get("steps", []):
            for action in step.get("action_records", []):
                color = "#e58f27" if action.get("kind") == "attack" else "#8d55b5" if action.get("kind") == "heal" else "#b8bec8"
                timeline.scatter(step.get("t", 0) - onset, lane, marker="|", s=80, color=color)
        timeline.text(0.01, lane + 0.18, method, transform=timeline.get_yaxis_transform(), fontsize=8)
    timeline.axvline(0, color="#222", linewidth=0.8)
    timeline.set_yticks(range(len(methods)), methods); timeline.set_xlabel("environment step relative to onset")
    timeline.set_title("SMAC action timeline (attack / heal / other)"); timeline.grid(axis="x", alpha=0.2)
    summary_path = root / "seed_summary.csv"
    summary_rows = list(csv.DictReader(summary_path.open(newline="", encoding="utf-8"))) if summary_path.exists() else []
    for x, metric, title in ((0.25, "focus_fire_completion_rate", "completion rate"), (0.75, "median_time_to_kill", "median time-to-kill")):
        for index, method in enumerate(methods):
            values = [float(r[metric]) for r in summary_rows if r.get("method") == method and r.get(metric) not in ("", "nan", "NaN")]
            if not values: continue
            xpos = x + (index - (len(methods) - 1) / 2) * 0.06
            low, high = bootstrap(values, int(hashlib.sha256(f"{method}-{metric}".encode()).hexdigest()[:8], 16))
            stats_ax.scatter([xpos] * len(values), values, label=method if x == 0.25 else None)
            stats_ax.errorbar([xpos], [float(np.mean(values))], yerr=[[float(np.mean(values)) - low], [high - float(np.mean(values))]], fmt="none", color="black", capsize=4)
        stats_ax.text(x, -0.18, title, ha="center", transform=stats_ax.transAxes)
    stats_ax.set_xlim(0, 1); stats_ax.set_xticks([]); stats_ax.set_ylabel("value"); stats_ax.legend(fontsize=8)
    fig.suptitle(f"SMACv1 5m_vs_6m case study — seed {selection['training_seed']} episode {selection['episode_id']}")
    output = Path(args.output) if args.output else root / "smacv1_case_study.png"
    fig.savefig(output, dpi=220, bbox_inches="tight"); fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--output", required=True); collect_parser.add_argument("--map", default="5m_vs_6m")
    collect_parser.add_argument("--episodes", type=int, default=100); collect_parser.add_argument("--episode-seed-base", type=int, default=DEFAULT_EPISODE_SEED_BASE)
    collect_parser.add_argument("--model", action="append", default=[], help="method,training_seed,checkpoint[,config.json]")
    collect_parser.add_argument("--config-json", action="append", default=[], help="method=/path/config.json")
    collect_parser.add_argument("--use-cuda", action="store_true"); collect_parser.add_argument("--no-frames", action="store_true"); collect_parser.add_argument("--overwrite", action="store_true")
    collect_parser.add_argument("--native-rgb", action="store_true",
                                help="request textured RGB frames from the SC2 client (requires EGL/OSMesa)")
    analyse_parser = sub.add_parser("analyse"); analyse_parser.add_argument("--data", required=True); analyse_parser.add_argument("--horizon", type=int, default=8)
    analyse_parser.add_argument("--selection-mode", choices=("complex", "representative"), default="complex",
                                help="select the richest one-sided paired engagement (default) or the old median-TTK case")
    plot_parser = sub.add_parser("plot"); plot_parser.add_argument("--data", required=True); plot_parser.add_argument("--horizon", type=int, default=8); plot_parser.add_argument("--output")
    args = parser.parse_args()
    if args.command == "collect": collect(args)
    elif args.command == "analyse": analyse(args)
    else: plot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
