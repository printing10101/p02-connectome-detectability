"""身体-大脑协同进化实验运行器.

基于已有 runner.py 的架构, 扩展支持:
- EmbodiedGA (权重 + 形态联合进化)
- 形态感知的评估 (evaluate_embodied_episode)
- 形态进化轨迹记录
- 身体/大脑进化开关 (用于对照实验)

用法:
    python scripts/run_embodied.py --exp mvp --pop 30 --gens 20
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from flyevo.brain import BrainConfig, LIFBrain
from flyevo.agent import SensorimotorMapping
from flyevo import world as W
from flyevo.embodiment import (
    EmbodiedGA, Morphology, MORPH_NAMES, MORPH_DEFAULTS,
    evaluate_embodied_episode, morphology_to_world_config,
)
from flyevo import viz
from flyevo.cascade import cascaded_evaluate, normalize_schedule, schedule_cost
from flyevo.runner import write_cascade_summary

ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# 并行 worker
# ============================================================================

_G: dict = {}


def _worker_init(n_nodes, pre, post, sign, nodes_records, world_cfg_dict, w0=None):
    _G["brain"] = LIFBrain(n_nodes, pre, post, sign, BrainConfig(), w0=w0)
    _G["mapping"] = SensorimotorMapping.from_nodes(pd.DataFrame(nodes_records))
    _G["world_cfg"] = W.WorldConfig(**world_cfg_dict)


def _eval_task(args):
    (weights, morph_vec, world_cfg_dict, env_seed, noise_seed,
     record_every, want_activity) = args
    from flyevo.embodiment import Morphology, evaluate_embodied_episode
    morph = Morphology.from_vector(morph_vec)
    world_cfg = W.WorldConfig(**world_cfg_dict)
    _G["brain"].set_weights(weights)
    stats, frames, activity = evaluate_embodied_episode(
        _G["brain"], _G["mapping"], world_cfg, weights, morph,
        env_seed, noise_seed,
        record_every=record_every, track_activity=want_activity,
    )
    return stats, frames if record_every else None, activity


# ============================================================================
# 数据加载
# ============================================================================

def load_connectome():
    nodes = pd.read_csv(ROOT / "data" / "flywire_subcircuit_nodes.csv")
    edges = pd.read_csv(ROOT / "data" / "flywire_subcircuit_edges.csv")
    w_init = np.log1p(edges["syn_count"].to_numpy(float))
    w_init /= np.median(w_init)
    return nodes, edges, w_init


# ============================================================================
# 适应度函数
# ============================================================================

def default_fitness(stats: dict) -> float:
    """默认适应度: 觅食效率 + 存活 + 能量管理.

    食物是主要奖励, 存活时间和最终能量是辅助.
    脑能耗已经在 final_energy_after_brain 中扣除.
    """
    food = stats.get("food", 0)
    alive = stats.get("alive_frac", 0)
    energy = stats.get("final_energy_after_brain", 0)
    toxin = stats.get("toxin_s", 0)
    # 食物为主, 存活和能量为辅, 毒物惩罚
    return food * 10.0 + alive * 5.0 + energy * 0.1 - toxin * 2.0


# ============================================================================
# 主实验循环
# ============================================================================

def run_embodied_experiment(
    exp_name: str = "mvp",
    pop_size: int = 30,
    n_gens: int = 20,
    duration: float = 15.0,
    n_eps: int = 2,
    seed: int = 1,
    evolve_body: bool = True,
    evolve_brain: bool = True,
    workers: int = 1,
    fitness_fn=None,
    world_cfg_overrides: dict | None = None,
    out_dir: str | None = None,
    cascade: bool = False,
    cascade_schedule: list[tuple[float, int]] | None = None,
    cascade_audit: bool = False,
    cascade_min_keep: int = 4,
    nt_policy: str = "gaba-only",
    edge_weight: str = "unit",
) -> Path:
    """运行身体-大脑协同进化实验.

    Args:
        exp_name: 实验名称 (用于输出目录)
        pop_size: 种群大小
        n_gens: 进化代数
        duration: 每回合时长(秒)
        n_eps: 每个个体评估的环境数
        seed: 随机种子
        evolve_body: 是否允许身体进化 (False=固定野生型身体)
        evolve_brain: 是否允许大脑进化 (False=固定野生型权重)
        workers: 并行worker数
        fitness_fn: 自定义适应度函数 (None=用default_fitness)
        world_cfg_overrides: 世界配置覆盖
        out_dir: 输出目录
    """
    if fitness_fn is None:
        fitness_fn = default_fitness

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) if out_dir else ROOT / "runs" / f"embodied_{exp_name}_s{seed}_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config = {
        "exp_name": exp_name, "pop_size": pop_size, "n_gens": n_gens,
        "duration": duration, "n_eps": n_eps, "seed": seed,
        "evolve_body": evolve_body, "evolve_brain": evolve_brain,
        "workers": workers, "world_cfg_overrides": world_cfg_overrides or {},
        "cascade": bool(cascade),
        "cascade_schedule": cascade_schedule if cascade else None,
        "cascade_audit": bool(cascade_audit) if cascade else None,
        "cascade_min_keep": cascade_min_keep if cascade else None,
        "cascade_effective_schedule": (
            normalize_schedule(cascade_schedule, n_eps, pop_size, cascade_min_keep)
            if cascade else None
        ),
        "nt_policy": nt_policy,
        "edge_weight": edge_weight,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    # 加载数据
    nodes, edges, w_init = load_connectome()
    pre_idx = edges["pre_idx"].to_numpy()
    post_idx = edges["post_idx"].to_numpy()
    # 递质符号基质 (P1.4): 默认 (gaba-only, unit) 与 v1 逐位一致
    from flyevo.nt_sign import load_substrate

    sign_arr, w0_arr, nt_meta = load_substrate(nt_policy, edge_weight)

    # 世界配置
    base_world = W.WorldConfig(duration=duration, predator_speed=0.0)
    if world_cfg_overrides:
        for k, v in world_cfg_overrides.items():
            setattr(base_world, k, v)
    world_dict = {k: v for k, v in base_world.__dict__.items()}

    # 创建进化算法
    ga = EmbodiedGA(
        w_init, pop_size=pop_size, seed=seed,
        evolve_body=evolve_body, evolve_brain=evolve_brain,
    )

    # 初始化worker
    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(len(nodes), pre_idx, post_idx, sign_arr,
                  nodes.to_dict("records"), world_dict, w0_arr),
    ) if workers > 1 else None
    if workers == 1:
        _worker_init(len(nodes), pre_idx, post_idx, sign_arr,
                     nodes.to_dict("records"), world_dict, w0_arr)

    # 记录
    gen_log = []
    morph_trace = []  # 每代种群平均形态
    best_morph_trace = []  # 每代最优个体形态
    events_log = []
    cascade_runs: list[dict] = []  # 级联模式: 每代的成本与保真度记录

    try:
        pbar = tqdm(range(n_gens), desc=f"[{exp_name}]", unit="gen")
        for gen in pbar:
            env_seed_base = seed * 1_000_000 + gen * 1000
            payload_cache: dict[int, tuple] = {}

            def _payload_of(i: int):
                p = payload_cache.get(i)
                if p is None:
                    p = (ga.get_weights(i), ga.get_morphology(i).to_vector())
                    payload_cache[i] = p
                return p

            def _eval_pairs(pairs, _base=env_seed_base):
                """按 (个体, 回合) 对评估。同一 (i,k) 的回合参数与全量协议完全一致,
                这是级联「嵌套前缀」性质成立的前提。"""
                tasks = [
                    (
                        *_payload_of(i), world_dict,
                        _base + k * 777,
                        _base * 97 + i + k * 13,
                        0, False,
                    )
                    for (i, k) in pairs
                ]
                if executor:
                    res = list(executor.map(_eval_task, tasks, chunksize=2))
                else:
                    res = [_eval_task(t) for t in tasks]
                return [r[0] for r in res]

            if cascade:
                cres = cascaded_evaluate(
                    _eval_pairs, fitness_fn, pop_size, n_eps,
                    schedule=cascade_schedule, audit=cascade_audit,
                    min_keep=cascade_min_keep,
                )
                stats_per_indiv, fits = cres.stats_per_indiv, cres.fits
                cascade_fields = cres.log_fields()
                cascade_runs.append({
                    "stage": 0, "gen": gen,
                    "episodes_spent": int(cres.episodes_spent),
                    "baseline_episodes": int(cres.baseline_episodes),
                    "saving": round(float(cres.saving), 4),
                    "n_promoted": int(cres.promoted.sum()),
                    "levels": cres.levels,
                    "audit": cres.audit,
                })
            else:
                # 保持原有任务顺序 (i 外 k 内), 让历史 run 逐位可复现
                pairs = [(i, k) for i in range(pop_size) for k in range(n_eps)]
                raw = _eval_pairs(pairs)
                stats_per_indiv = []
                for i in range(pop_size):
                    eps_stats = [raw[i * n_eps + k] for k in range(n_eps)]
                    avg = {k: float(np.mean([s[k] for s in eps_stats])) for k in eps_stats[0]}
                    stats_per_indiv.append(avg)
                fits = np.array([fitness_fn(s) for s in stats_per_indiv])
                cascade_fields = {}

            # 记录
            mean_morph = ga.mean_morphology()
            best_idx, best_w, best_morph = ga.best(fits)
            row = {
                "gen": gen,
                "best_fit": float(fits.max()),
                "mean_fit": float(fits.mean()),
                "std_fit": float(fits.std()),
                "mean_food": float(np.mean([s["food"] for s in stats_per_indiv])),
                "best_food": float(stats_per_indiv[best_idx]["food"]),
                "mean_alive": float(np.mean([s["alive_frac"] for s in stats_per_indiv])),
                "mean_body_size": float(mean_morph.body_size),
                "mean_wing_size": float(mean_morph.wing_size),
                "mean_brain_energy": float(mean_morph.brain_energy),
                "mean_sens_gain": float(mean_morph.sens_gain),
                "best_body_size": float(best_morph.body_size),
                "best_wing_size": float(best_morph.wing_size),
                "best_brain_energy": float(best_morph.brain_energy),
                "best_can_fly": int(best_morph.can_fly()),
                "diversity_w": ga.diversity()["weight_div"],
                "diversity_m": ga.diversity()["morph_div"],
                **cascade_fields,
            }
            gen_log.append(row)
            morph_trace.append(mean_morph.to_vector())
            best_morph_trace.append(best_morph.to_vector())

            pbar.set_postfix(
                best=f"{fits.max():.1f}",
                mean=f"{fits.mean():.1f}",
                food=f"{row['mean_food']:.1f}",
                body=f"{row['mean_body_size']:.2f}",
            )

            # 进化
            events = ga.evolve(fits)
            events_log.append(events)

        # ---- 实验收尾: 回放最优个体 ----
        best_idx, best_w, best_morph = ga.best(fits)
        replay_seed = seed * 1_000_000 + 999
        replay_task = (
            best_w, best_morph.to_vector(), world_dict,
            replay_seed, replay_seed + 1, 5, True,
        )
        if executor:
            best_stats, frames, best_activity = executor.submit(_eval_task, replay_task).result()
        else:
            best_stats, frames, best_activity = _eval_task(replay_task)

        if frames:
            viz.render_replay(frames, base_world, out / "replay_best.gif")

    finally:
        if executor:
            executor.shutdown()

    # ---- 保存结果 ----
    df = pd.DataFrame(gen_log)
    df.to_csv(out / "gen_log.csv", index=False)

    if cascade and cascade_runs:
        write_cascade_summary(out, cascade_runs, pop_size, n_eps,
                              cascade_schedule, cascade_min_keep)

    np.save(out / "morph_trace_mean.npy", np.array(morph_trace))
    np.save(out / "morph_trace_best.npy", np.array(best_morph_trace))

    # 保存最优个体
    np.save(out / "best_weights.npy", best_w)
    (out / "best_morphology.json").write_text(
        json.dumps(best_morph.to_dict(), indent=2, ensure_ascii=False)
    )

    # 可视化
    try:
        _plot_fitness(df, out / "fitness.png")
        _plot_morph_evolution(df, out / "morph_evolution.png")
    except Exception as e:
        (out / "viz_error.txt").write_text(str(e))

    # 写报告
    _write_report(out, exp_name, config, df, best_stats, best_morph, morph_trace)

    return out


def _plot_fitness(df: pd.DataFrame, path: Path):
    """绘制适应度进化曲线 (不依赖viz模块的stage列)."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(df["gen"], df["best_fit"], label="Best", linewidth=1.5, color="#d62728")
    ax1.plot(df["gen"], df["mean_fit"], label="Mean", linewidth=1.5, color="#1f77b4")
    ax1.fill_between(df["gen"], df["mean_fit"] - df["std_fit"],
                     df["mean_fit"] + df["std_fit"], alpha=0.2, color="#1f77b4")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.set_title("适应度进化")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["gen"], df["mean_food"], label="Mean Food", linewidth=1.5, color="#2ca02c")
    ax2.plot(df["gen"], df["best_food"], label="Best Food", linewidth=1.5, color="#ff7f0e")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Food eaten")
    ax2.set_title("觅食能力进化")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def _plot_morph_evolution(df: pd.DataFrame, path: Path):
    """绘制形态参数进化曲线."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    params = [
        ("mean_body_size", "体型", "body_size"),
        ("mean_wing_size", "翅大小", "wing_size"),
        ("mean_brain_energy", "脑能耗占比", "brain_energy"),
        ("mean_sens_gain", "感觉增益", "sens_gain"),
        ("best_body_size", "最优体型", "body_size"),
        ("best_wing_size", "最优翅大小", "wing_size"),
    ]
    for ax, (col, label, _) in zip(axes.flat, params):
        ax.plot(df["gen"], df[col], label=label, linewidth=1.5)
        ax.set_xlabel("Generation")
        ax.set_ylabel(label)
        ax.set_title(f"{label} 进化轨迹")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def _write_report(out, exp_name, config, df, best_stats, best_morph, morph_trace):
    """写实验报告."""
    first5 = df.head(5)
    last5 = df.tail(5)
    lines = [
        f"# 身体-大脑协同进化实验报告: {exp_name}",
        "",
        f"- 种子: {config['seed']} | 种群: {config['pop_size']} | 代数: {config['n_gens']}",
        f"- 身体进化: {'✅ 开启' if config['evolve_body'] else '❌ 关闭(固定)'}",
        f"- 大脑进化: {'✅ 开启' if config['evolve_brain'] else '❌ 关闭(固定)'}",
        f"- 神经元: 4,409 (FlyWire 嗅觉通路子回路)",
        "",
        "## 适应度进化",
        "",
        f"- 前5代 mean_fit: {first5['mean_fit'].mean():.1f}",
        f"- 后5代 mean_fit: {last5['mean_fit'].mean():.1f}",
        f"- 增益: {last5['mean_fit'].mean() - first5['mean_fit'].mean():+.1f}",
        f"- 最优适应度: {df['best_fit'].max():.1f} (第{df.loc[df['best_fit'].idxmax(), 'gen']}代)",
        "",
        "## 觅食行为进化",
        "",
        f"- 前5代 mean_food: {first5['mean_food'].mean():.2f}",
        f"- 后5代 mean_food: {last5['mean_food'].mean():.2f}",
        f"- 最优个体觅食: {best_stats.get('food', 0)}",
        "",
        "## 形态进化 (种群平均)",
        "",
    ]
    morph_start = morph_trace[0]
    morph_end = morph_trace[-1]
    for name, s, e in zip(MORPH_NAMES, morph_start, morph_end):
        lines.append(f"- {name}: {s:.3f} → {e:.3f} ({e-s:+.3f})")

    lines += [
        "",
        "## 最优个体形态",
        "",
    ]
    for k, v in best_morph.to_dict().items():
        lines.append(f"- {k}: {v}")

    lines += [
        "",
        "## 最优个体回放统计",
        "",
    ]
    for k, v in best_stats.items():
        lines.append(f"- {k}: {v}")

    lines += [
        "",
        "## 多样性变化",
        "",
        f"- 权重多样性: {df['diversity_w'].iloc[0]:.4f} → {df['diversity_w'].iloc[-1]:.4f}",
        f"- 形态多样性: {df['diversity_m'].iloc[0]:.4f} → {df['diversity_m'].iloc[-1]:.4f}",
        "",
        "## 结论",
        "",
    ]
    fit_gain = last5["mean_fit"].mean() - first5["mean_fit"].mean()
    food_gain = last5["mean_food"].mean() - first5["mean_food"].mean()
    if fit_gain > 2.0 and food_gain > 0.2:
        lines.append("✅ **进化发生**: 适应度和觅食能力均显著提升.")
    elif fit_gain > 0:
        lines.append("⚠️ **弱进化**: 适应度有提升但不显著, 可能需要更多代数或调整选择压力.")
    else:
        lines.append("❌ **未观察到进化**: 适应度无提升, 需要检查适应度函数或进化参数.")

    if config["evolve_body"]:
        body_changed = np.abs(morph_end - morph_start).sum()
        lines.append(f"\n形态总变化量: {body_changed:.3f}")
        if body_changed > 0.5:
            lines.append("✅ **身体进化发生**: 形态参数发生了可测量的变化.")
        else:
            lines.append("⚠️ **身体进化微弱**: 形态参数变化不大, 可能选择压力不足以驱动身体进化.")

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
