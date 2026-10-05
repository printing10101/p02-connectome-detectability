"""可视化: 适应度曲线, 行为回放 GIF, 权重变化清单."""
from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_fitness(gen_log: pd.DataFrame, out_path: Path) -> None:
    stages = sorted(gen_log["stage"].unique())
    n = len(stages)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.6 * nrows), squeeze=False)
    for ax, st in zip(axes.flat, stages):
        d = gen_log[gen_log["stage"] == st]
        x = np.arange(len(d))
        ax.fill_between(x, d["mean_fit"] - d["std_fit"], d["mean_fit"] + d["std_fit"],
                        color="teal", alpha=0.2, label="±std")
        ax.plot(x, d["mean_fit"], color="teal", label="mean")
        ax.plot(x, d["best_fit"], color="crimson", linestyle="--", alpha=0.7, label="best")
        ax.set_title(f"Stage {st}")
        ax.set_xlabel("generation")
        ax.set_ylabel("fitness")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def render_replay(
    frames: list,
    cfg,
    out_path: Path,
    fps: int = 20,
) -> None:
    """把 evaluate_episode 记录的帧渲染成 GIF."""
    xs = np.linspace(0, cfg.width, 120)
    ys = np.linspace(0, cfg.height, 120)
    gx, gy = np.meshgrid(xs, ys)
    field = np.zeros_like(gx)
    for p in frames[0][2]:
        field += np.exp(-((gx - p[0]) ** 2 + (gy - p[1]) ** 2) / (2 * cfg.food_odor_sigma**2))

    imgs = []
    trail = []
    for pos, heading, food, toxin, pred, energy in frames[::2]:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(field, origin="lower", extent=[0, cfg.width, 0, cfg.height],
                  cmap="Greens", alpha=0.35)
        ax.scatter(food[:, 0], food[:, 1], s=14, c="seagreen", zorder=3)
        if len(toxin):
            for tp in toxin:
                ax.add_patch(plt.Circle(tp, cfg.toxin_radius, color="purple", alpha=0.5, zorder=2))
        if pred is not None:
            ax.scatter(pred[0], pred[1], s=60, c="crimson", marker="X", zorder=4)
        trail.append(pos)
        tr = np.array(trail[-60:])
        ax.plot(tr[:, 0], tr[:, 1], color="black", alpha=0.4, lw=1, zorder=4)
        ax.annotate("", xy=pos + 3 * np.array([np.cos(heading), np.sin(heading)]), xytext=pos,
                    arrowprops=dict(arrowstyle="->", color="black"), zorder=5)
        ax.set_xlim(0, cfg.width)
        ax.set_ylim(0, cfg.height)
        ax.set_title(f"t={len(trail)*cfg.dt_ctrl*2:.1f}s  E={energy:.0f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        imgs.append(img)
        plt.close(fig)

    imageio.mimsave(out_path, imgs, fps=fps)


def plot_neuron_influence(tab: pd.DataFrame, cfg, out_path: Path) -> None:
    """阶段前后每神经元的下行影响力: 散点(按细胞群上色) + 分群漂移."""
    groups = ["ORN", "PN", "LH", "DN"]
    colors = {"ORN": "#4c72b0", "PN": "#55a868", "LH": "#c44e52", "DN": "#8172b3"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    lim = max(tab["out0"].abs().max(), tab["out1"].abs().max()) * 1.05 + 1e-9
    ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.8, linestyle=":")
    for g in groups:
        d = tab[tab["group"] == g]
        ax.scatter(d["out0"], d["out1"], s=6, alpha=0.45, color=colors[g], label=g)
    ax.set_xlabel("out-influence at stage start")
    ax.set_ylabel("out-influence at stage end")
    ax.set_title("per-neuron influence drift (dot on y=x = unchanged)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    means = [tab[tab["group"] == g]["delta_out"].mean() for g in groups]
    stds = [tab[tab["group"] == g]["delta_out"].std() / np.sqrt(len(tab[tab["group"] == g])) for g in groups]
    ax.bar(groups, means, yerr=stds, color=[colors[g] for g in groups], alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("mean Δ out-influence (population)")
    ax.set_title("which cell groups did selection push?")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_neuron_traces(
    tab: pd.DataFrame,
    snaps: np.ndarray,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
    sign: np.ndarray,
    out_path: Path,
    top_k: int = 8,
) -> None:
    """改动最大的 top_k 神经元的下行影响力代际轨迹."""
    T, _ = snaps.shape
    traces = np.zeros((T, len(tab)))
    for t in range(T):
        traces[t] = np.bincount(
            pre_idx, weights=snaps[t] * sign, minlength=len(tab)
        )
    top_idx = tab["delta_out"].abs().sort_values(ascending=False).index[:top_k]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i in top_idx:
        name = str(tab.loc[i, "cell_type"]) if pd.notna(tab.loc[i, "cell_type"]) else "unnamed"
        ax.plot(traces[:, i], label=f"{tab.loc[i,'group']}/{name}", lw=1.4)
    ax.set_xlabel("generation (population mean genome)")
    ax.set_ylabel("out-influence")
    ax.set_title("neuron-level evolutionary trajectories (top movers)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_edge_fixation(
    registry: pd.DataFrame,
    freq_matrix: np.ndarray,
    fix: dict,
    nodes: pd.DataFrame,
    out_path: Path,
) -> None:
    """边存在频率的代际轨迹: 固定的新连接(实线) + 被剪除的真实连接(虚线)."""
    from .structure_evo import _cell_name

    fig, ax = plt.subplots(figsize=(9, 5))
    for eid in fix["fixed_new_ids"][:8]:
        r = registry.loc[eid]
        label = (f"+ {_cell_name(nodes, int(r['pre_idx']))}"
                 f" -> {_cell_name(nodes, int(r['post_idx']))}")
        ax.plot(np.nan_to_num(freq_matrix[:, eid], nan=0.0), lw=1.6, label=label)
    for eid in fix["pruned_real_ids"][:4]:
        r = registry.loc[eid]
        label = (f"- {_cell_name(nodes, int(r['pre_idx']))}"
                 f" -> {_cell_name(nodes, int(r['post_idx']))}")
        ax.plot(np.nan_to_num(freq_matrix[:, eid], nan=0.0), lw=1.2, linestyle="--",
                color="grey", alpha=0.7, label=label)
    ax.set_xlabel("generation")
    ax.set_ylabel("edge presence frequency in population")
    ax.set_title("structural evolution: new edges fixed / real edges pruned")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def weight_diff_report(
    nodes: pd.DataFrame,
    pre_u: np.ndarray,
    post_u: np.ndarray,
    sign_u: np.ndarray,
    genome_start: np.ndarray,
    genome_end: np.ndarray,
    out_csv: Path,
    top_k: int = 20,
    origin: np.ndarray | None = None,
    added_gen: np.ndarray | None = None,
    syn_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """阶段前后平均基因组的差异, 列出被进化改动最大的连接(带细胞类型名).

    pre_u/post_u/sign_u 为当前边宇宙; origin/added_gen 标注边来源(结构模式).
    """
    diff = np.asarray(genome_end) - np.asarray(genome_start)
    n = len(diff)
    df = pd.DataFrame({
        "pre_idx": np.asarray(pre_u)[:n].astype(int),
        "post_idx": np.asarray(post_u)[:n].astype(int),
        "sign_arr": np.asarray(sign_u)[:n],
        "delta": diff,
        "abs_delta": np.abs(diff),
    })
    if origin is None:
        df["origin"] = "real"
    else:
        df["origin"] = np.where(np.asarray(origin)[:n] == 1, "added", "real")
    if added_gen is not None:
        df["added_gen"] = np.asarray(added_gen)[:n]

    if syn_table is not None and len(syn_table):
        first = syn_table.drop_duplicates(["pre_idx", "post_idx"]).set_index(
            ["pre_idx", "post_idx"]
        )
        keys = list(zip(df["pre_idx"], df["post_idx"]))
        df["syn_count"] = [first["syn_count"].get(k, np.nan) for k in keys]

    top = df.nlargest(top_k, "abs_delta")
    out = top.assign(
        pre_type=nodes.loc[top["pre_idx"], "cell_type"].to_numpy(),
        pre_group=nodes.loc[top["pre_idx"], "group"].to_numpy(),
        post_type=nodes.loc[top["post_idx"], "cell_type"].to_numpy(),
        post_group=nodes.loc[top["post_idx"], "group"].to_numpy(),
    )[
        ["pre_group", "pre_type", "post_group", "post_type", "origin",
         "syn_count", "delta"]
    ]
    out.to_csv(out_csv, index=False)
    return out
