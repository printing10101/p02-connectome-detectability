"""统一读数: 结构新颖度 SNS / 固定率对照 / 方言指数.

与 GlossoGen 的 perplexity 对齐: 度量「离初始硬件语言有多远」.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def structural_novelty_score(
    fixed_new_edges: Iterable[tuple[int, int]],
    real_edge_set: set[tuple[int, int]],
    neutral_freq: dict[tuple[int, int], float] | None = None,
    eps: float = 0.05,
) -> dict:
    """SNS: 固定新边中「真实连接组没有、且中性里罕见」的比例.

    fixed_new_edges: (pre_idx, post_idx) 序列, 本代被固定的新增边
    real_edge_set: 原始 FlyWire 边集合
    neutral_freq: 同代中性对照里该边的种群存在频率 (可缺省)
    """
    edges = [(int(a), int(b)) for a, b in fixed_new_edges]
    n_fixed = len(edges)
    novel = []
    for e in edges:
        if e in real_edge_set:
            continue
        if neutral_freq is not None:
            if neutral_freq.get(e, 0.0) >= eps:
                continue
        novel.append(e)
    n_novel = len(novel)
    sns = (n_novel / n_fixed) if n_fixed else 0.0
    return {
        "n_fixed_new": n_fixed,
        "n_novel_fixed": n_novel,
        "sns": float(sns),
        "novel_edges": novel,
    }


def fixation_vs_neutral(
    real_counts: list[int],
    neutral_counts: list[int],
    q: float = 0.99,
) -> dict:
    """结构固定数 vs 中性 P99 带."""
    real = np.asarray(real_counts, dtype=float)
    neu = np.asarray(neutral_counts, dtype=float)
    p99 = float(np.quantile(neu, q)) if len(neu) else float("nan")
    mean_real = float(real.mean()) if len(real) else float("nan")
    mean_neu = float(neu.mean()) if len(neu) else float("nan")
    exceed = bool(len(real) and mean_real > p99)
    return {
        "mean_real": mean_real,
        "mean_neutral": mean_neu,
        "neutral_p99": p99,
        "exceeds_p99": exceed,
        "ratio_mean": (mean_real / mean_neu) if mean_neu else float("nan"),
    }


def dialect_jaccard(edges_a: set[tuple[int, int]], edges_b: set[tuple[int, int]]) -> float:
    """两个亚群固定边集合的 Jaccard; 1=同一方言, 0=完全分化."""
    if not edges_a and not edges_b:
        return 1.0
    inter = len(edges_a & edges_b)
    union = len(edges_a | edges_b)
    return inter / union if union else 1.0


def weight_selection_ratio(gain_real: float, gain_neutral: float) -> float:
    """选择/漂移倍率; 中性为负或零时返回 nan."""
    if gain_neutral <= 0:
        return float("nan")
    return float(gain_real / gain_neutral)


def pressure_boundary_from_ladder(
    ladder: list[dict],
) -> dict:
    """从压力阶梯结果里找权重平台与结构显形档.

    ladder: 每项至少含 level, weight_gain, sns, struct_exceeds_p99
    """
    if not ladder:
        return {"p_star": None, "weight_plateau_level": None}
    levels = sorted(ladder, key=lambda x: x["level"])
    # 权重平台: 相邻档权重增益相对增量 < 10%
    plateau = None
    for i in range(1, len(levels)):
        prev = levels[i - 1]["weight_gain"]
        cur = levels[i]["weight_gain"]
        if prev > 0 and (cur - prev) / prev < 0.10:
            plateau = levels[i - 1]["level"]
            break
    p_star = None
    for lv in levels:
        if lv.get("struct_exceeds_p99") or lv.get("sns", 0) >= 0.3:
            p_star = lv["level"]
            break
    return {
        "weight_plateau_level": plateau,
        "p_star": p_star,
        "ladder": levels,
    }
