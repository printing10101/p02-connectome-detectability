"""神经元层进化分析: 把边级基因组聚合成每个神经元的"表型".

核心量:
- 下行影响力 out = Σ(出边权重×递质符号): 这个神经元对下游的总驱动
- 上游驱动 in = Σ(入边权重×递质符号): 它被推得多狠
- 发放率 hz: 评估回合内每神经元的平均发放 (功能表型)

进化若发生在神经元粒度, 应看到: 影响力/发放率在代际轨迹上持续
定向漂移, 且集中于特定细胞群 -- 而不是全脑均匀噪声.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def neuron_strengths(
    genome_mean: np.ndarray,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
    n_nodes: int,
    sign: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """边级基因组 -> 每神经元的 (净下行影响力, 净上游驱动)."""
    w = genome_mean * sign
    out = np.bincount(pre_idx, weights=w, minlength=n_nodes)
    inn = np.bincount(post_idx, weights=w, minlength=n_nodes)
    return out, inn


def neuron_evo_table(
    nodes: pd.DataFrame,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
    sign: np.ndarray,
    genome_start: np.ndarray,
    genome_end: np.ndarray,
    hz_start: np.ndarray | None,
    hz_end: np.ndarray | None,
) -> pd.DataFrame:
    """组装神经元级进化表: 影响力与发放率的阶段前后对比."""
    n = len(nodes)
    out0, in0 = neuron_strengths(genome_start, pre_idx, post_idx, n, sign)
    out1, in1 = neuron_strengths(genome_end, pre_idx, post_idx, n, sign)
    tab = nodes[["root_id", "group", "cell_type", "side"]].copy()
    tab["out0"], tab["out1"] = out0, out1
    tab["delta_out"] = out1 - out0
    tab["in0"], tab["in1"] = in0, in1
    if hz_start is not None and hz_end is not None:
        tab["hz0"], tab["hz1"] = hz_start, hz_end
        tab["delta_hz"] = hz_end - hz_start
    return tab


def summarize_neuron_evo(tab: pd.DataFrame) -> dict:
    """提炼神经元级进化叙事: 征用/沉默/分群漂移/改动集中度."""
    d = tab["delta_out"]
    sigma = float(d.std()) or 1.0
    changed = tab[d.abs() > 2 * sigma]
    top = tab.reindex(d.abs().sort_values(ascending=False).index).head(10)

    summary = {
        "n_neurons": len(tab),
        "n_changed_2sigma": int(len(changed)),
        "frac_changed": float((d.abs() > 2 * sigma).mean()),
        "top_movers": [
            (r["group"], str(r["cell_type"]), float(r["delta_out"]))
            for _, r in top.iterrows()
        ],
        "group_shift": tab.groupby("group")["delta_out"].mean().to_dict(),
    }
    if "delta_hz" in tab.columns:
        recruited = tab[(tab["hz0"] < 1.0) & (tab["hz1"] > 10.0)]
        silenced = tab[(tab["hz0"] > 10.0) & (tab["hz1"] < 1.0)]
        summary["recruited"] = int(len(recruited))
        summary["silenced"] = int(len(silenced))
        summary["recruited_types"] = [
            f"{r['group']}/{r['cell_type']}" for _, r in recruited.head(5).iterrows()
        ]
    return summary
