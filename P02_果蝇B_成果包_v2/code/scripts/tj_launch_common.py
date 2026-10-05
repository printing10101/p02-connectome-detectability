"""TJ 系列启动器的公共件: 固定判定 analyze / ensure / A1 族判读。

风格沿用 verdict_c8_* 判决脚本 (COMMON + workers=7 + 跳过已有 run + verdict JSON);
抽出公共件是为了让四个启动器只差协议参数与判据键, 不必各抄 150 行 analyze。
判据一律经 flyevo.tj_criteria 从冻结 JSON 载入, 缺文件/缺键 = 拒绝运行。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from flyevo.runner import run_experiment
from flyevo.structure_evo import FIX_FREQ, sustained_above

ROOT = Path(__file__).resolve().parents[1]
NODES = pd.read_csv(ROOT / "data" / "flywire_subcircuit_nodes.csv")

USEFUL = ("ORN->DN", "ORN->LH", "PN->DN", "LH->DN")


def analyze(tag_dir: Path, sustain: int = 5, thresh: float = FIX_FREQ) -> dict:
    """与 run_c8_long.analyze 同款: 频率矩阵重建 -> origin==1 且 sustained_above 计固定.

    比 C8 版多返回 fixed_pairs (固定边的 (pre,post) 列表), 供 A2b 区分最优边/诱饵。
    """
    gl = pd.read_csv(tag_dir / "gen_log.csv")
    reg = pd.read_csv(tag_dir / "edge_registry.csv")
    npz = np.load(tag_dir / "edge_freq.npz")
    gens_a, eids_a, vals_a = npz["gens"], npz["eids"], npz["vals"]
    n_gen = int(gens_a.max()) + 1
    n_e = max(int(eids_a.max()) + 1, int(reg.index.max()) + 1 if len(reg) else 0)
    fm = np.zeros((n_gen, n_e))
    for g, e, v in zip(gens_a, eids_a, vals_a):
        if e < n_e:
            fm[int(g), int(e)] = float(v)

    fixed_flows: dict[str, int] = {}
    fixed_pairs: list[tuple[int, int]] = []
    for eid in reg.index:
        if int(eid) >= n_e or reg.loc[eid, "origin"] != 1:
            continue
        row = fm[:, int(eid)]
        if not sustained_above(row, thresh, sustain=sustain):
            continue
        p, q = int(reg.loc[eid, "pre_idx"]), int(reg.loc[eid, "post_idx"])
        fixed_pairs.append((p, q))
        fl = (
            f"{NODES.group.iloc[p]}->{NODES.group.iloc[q]}"
            if p < len(NODES) and q < len(NODES)
            else "?"
        )
        fixed_flows[fl] = fixed_flows.get(fl, 0) + 1

    n_fixed = len(fixed_pairs)
    n_useful = sum(v for k, v in fixed_flows.items() if k in USEFUL)
    return {
        "food": float(gl["mean_food"].iloc[-1]),
        "food_best_gen": float(gl["mean_food"].max()),
        "n_added": int((reg["origin"] == 1).sum()),
        "n_fixed": n_fixed,
        "n_useful": n_useful,
        "flows": dict(sorted(fixed_flows.items(), key=lambda x: -x[1])),
        "fixed_pairs": fixed_pairs,
    }


def ratio(real: int | None, neu: int | None) -> float | None:
    """real/neu 固定数比; 中性臂为 0 时用 inf/None 表达方向 (沿用 C8 脚本约定)."""
    if not neu:
        return float("inf") if real else None
    return real / neu


def ensure(out_root: Path, seed: int, kind: str, common: dict) -> tuple[Path, str]:
    """跑单个 run, 已有完整产物则跳过 (断点续批)。返回 (run 目录, 'done'|'skip')。"""
    tag = f"{kind}_s{seed}"
    out = out_root / tag
    if (out / "gen_log.csv").exists() and (out / "edge_registry.csv").exists():
        print("SKIP", tag, flush=True)
        return out, "skip"
    ctrl = "none" if kind == "real" else "neutral"
    t0 = time.time()
    print("START", tag, flush=True)
    run_experiment(seed=seed, control=ctrl, out_dir=str(out), **common)
    print("DONE", tag, "min", round((time.time() - t0) / 60, 1), flush=True)
    return out, "done"


def structure_verdict(ratios: list, crit: dict) -> tuple[str, dict]:
    """A1 族结构固定率判读: 冻结判据只有两个数 —— 阈值与支持种子数。

    n_over >= support             -> SUPPORTED (阴性叙事被打破)
    n_over <= n_seeds - support   -> INSENSITIVE (剩余种子全中也到不了支持线, 稳健阴性)
    其余                          -> FRAGMENTARY (部分种子显形)
    """
    thresh = float(crit["fixed_ratio_threshold"])
    support = int(crit["seeds_over_threshold_for_support"])
    vals = [r for r in ratios if r is not None]
    n_over = sum(1 for r in vals if r >= thresh)
    n = len(ratios)
    stats = {
        "ratio_threshold": thresh,
        "support_needed": support,
        "n_seeds": n,
        "n_seeds_over_threshold": n_over,
        "ratio_mean": float(np.mean(vals)) if vals else None,
    }
    if n_over >= support:
        return "SUPPORTED", stats
    if n_over <= n - support:
        return "INSENSITIVE", stats
    return "FRAGMENTARY", stats


def paired_gain_ci(real_foods: list, neu_foods: list) -> dict:
    """配对行为增益的 t 区间 (A2 manifest 判据「行为层增益 CI 不含 0」)。

    环境里有 scipy, 但批量机不装重依赖更稳, 用查表 t 分位 (df = n-1, 双侧 0.05)。
    """
    d = np.asarray(real_foods, float) - np.asarray(neu_foods, float)
    n = len(d)
    if n == 1:  # 单种子 (冒烟) 下 CI 无定义: 置宽区间, 判「不含 0」为假
        return {"mean": float(d[0]), "se": float("nan"),
                "ci_low": float("-inf"), "ci_high": float("inf"),
                "excludes_zero": False}
    # 11-19 为 A3 (seeds 1-20, df=n-1) 补的档; n>=21 用正态近似 z=1.96 (偏窄, 留痕即可)
    t_crit = {2: 12.706, 3: 4.303, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
              11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
              16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093}.get(n, 1.96)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "mean": mean,
        "se": se,
        "ci_low": mean - t_crit * se,
        "ci_high": mean + t_crit * se,
        "excludes_zero": bool(mean - t_crit * se > 0 or mean + t_crit * se < 0),
    }
