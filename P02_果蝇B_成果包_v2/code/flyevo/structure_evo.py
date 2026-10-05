"""结构进化分析: 从边频率轨迹里提取固定/剪除/重复谱系事件.

核心定义:
- 新增边"固定": 存在频率 >=0.8 且之后保持 >=10 代 (被选择接受的新连接)
- 真实边"剪除": 存在频率 <=0.2 且之后保持 >=10 代 (被选择丢弃的旧连接)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIX_FREQ = 0.8
DEL_FREQ = 0.2
SUSTAIN = 10
# 注意: added_gen 一刀切剔除已弃用作主指标 (seed2 诊断: 有用边可在 gen1-3 固定).
# 主判决改用跨条件独有固定, 见 analyze_fixation_cross.
EARLY_DRIFT_GEN = 3
EARLY_DRIFT_FREQ = 0.5


def freq_snapshot_to_long(freqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每代的频率向量列表 -> (gens, edge_ids, freqs) 长表."""
    gens, eids, vals = [], [], []
    for g, f in enumerate(freqs):
        n = len(f)
        gens.append(np.full(n, g, dtype=np.int32))
        eids.append(np.arange(n, dtype=np.int32))
        vals.append(f.astype(np.float32))
    return (np.concatenate(gens), np.concatenate(eids), np.concatenate(vals))


def sustained_below(freq_row: np.ndarray, thresh: float, sustain: int = SUSTAIN) -> bool:
    """频率在最后 sustain 代内是否持续低于阈值."""
    tail = freq_row[-sustain:]
    return len(tail) >= sustain and bool((tail <= thresh).all())


def sustained_above(freq_row: np.ndarray, thresh: float, sustain: int = SUSTAIN) -> bool:
    tail = freq_row[-sustain:]
    return len(tail) >= sustain and bool((tail >= thresh).all())


def analyze_fixation(
    registry: pd.DataFrame,
    freq_matrix: np.ndarray,   # (n_snapshots, universe_size_at_end) 对齐到最终宇宙
    gens: list[int],
    exclude_early_drift: bool = False,
    early_gen: int = EARLY_DRIFT_GEN,
    early_freq: float = EARLY_DRIFT_FREQ,
    sustain: int = SUSTAIN,
) -> dict:
    """从对齐后的频率矩阵提取固定/剪除事件.

    exclude_early_drift 默认 False: 早期固定可能是真选择+精英克隆,
    主判决请用 analyze_fixation_cross (跨条件独有).
    """
    added = registry["origin"] == 1
    fixed_new, pruned_real, drift_artifact = [], [], []
    n_cols = freq_matrix.shape[1] if freq_matrix.ndim == 2 else 0
    for eid in registry.index:
        if int(eid) >= n_cols:
            continue
        row = freq_matrix[:, int(eid)]
        if added.loc[eid] and sustained_above(row, FIX_FREQ, sustain=sustain):
            if exclude_early_drift:
                ag = registry.loc[eid].get("added_gen", -1)
                if pd.isna(ag) or int(ag) < 0 or int(ag) <= early_gen:
                    drift_artifact.append(eid)
                    continue
            fixed_new.append(eid)
        if (not added.loc[eid]) and sustained_below(row, DEL_FREQ, sustain=sustain):
            pruned_real.append(eid)
    return {
        "fixed_new_ids": fixed_new,
        "pruned_real_ids": pruned_real,
        "drift_artifact_ids": drift_artifact,
        "n_fixed": len(fixed_new),
        "n_pruned": len(pruned_real),
        "n_drift_artifact": len(drift_artifact),
        "gens": gens,
    }


def _edge_keys(registry: pd.DataFrame, eids) -> set[tuple[int, int]]:
    keys = set()
    for eid in eids:
        p = int(registry.loc[eid, "pre_idx"])
        q = int(registry.loc[eid, "post_idx"])
        keys.add((p, q))
    return keys


def _flow_of(nodes: pd.DataFrame | None, pre: int, post: int) -> str:
    if nodes is None:
        return "?"
    if pre < len(nodes) and post < len(nodes):
        return f"{nodes.iloc[pre]['group']}->{nodes.iloc[post]['group']}"
    return "?"


def analyze_fixation_cross(
    reg_real: pd.DataFrame,
    fm_real: np.ndarray,
    reg_neu: pd.DataFrame,
    fm_neu: np.ndarray,
    nodes: pd.DataFrame | None = None,
    thresh: float = FIX_FREQ,
    sustain: int = 5,
    useful_flows: tuple[str, ...] = (
        "ORN->DN", "ORN->LH", "PN->DN", "LH->DN", "ORN->PN", "PN->LH",
    ),
) -> dict:
    """跨条件独有固定 — C8 主判决协议.

    同 seed 时真实/中性共享突变流, 共有高频边 = 漂移/共有突变;
    仅在真实高频 = 候选选择固定 (onlyR); 仅在中性高频 = onlyN.
    """
    fix_r = analyze_fixation(reg_real, fm_real, list(range(fm_real.shape[0])),
                             exclude_early_drift=False, sustain=sustain)
    # 用 thresh 而不是全局 FIX_FREQ 时, 直接扫
    def high_keys(reg, fm):
        keys = {}
        n_cols = fm.shape[1]
        for eid in reg.index:
            if int(eid) >= n_cols or reg.loc[eid, "origin"] != 1:
                continue
            row = fm[:, int(eid)]
            if sustained_above(row, thresh, sustain=sustain):
                keys[(int(reg.loc[eid, "pre_idx"]), int(reg.loc[eid, "post_idx"]))] = float(
                    reg.loc[eid, "final_freq"]
                )
        return keys

    kR = high_keys(reg_real, fm_real)
    kN = high_keys(reg_neu, fm_neu)
    only_r = set(kR) - set(kN)
    only_n = set(kN) - set(kR)
    shared = set(kR) & set(kN)

    def flow_counts(keys):
        c: dict[str, int] = {}
        for p, q in keys:
            f = _flow_of(nodes, p, q)
            c[f] = c.get(f, 0) + 1
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    only_r_flows = flow_counts(only_r)
    only_n_flows = flow_counts(only_n)
    n_useful_r = sum(v for k, v in only_r_flows.items() if k in useful_flows)
    n_useful_n = sum(v for k, v in only_n_flows.items() if k in useful_flows)

    return {
        "thresh": thresh,
        "sustain": sustain,
        "n_high_real": len(kR),
        "n_high_neu": len(kN),
        "n_only_real": len(only_r),
        "n_only_neu": len(only_n),
        "n_shared": len(shared),
        "ratio_only": (len(only_r) / len(only_n)) if only_n else (float("inf") if only_r else None),
        "only_real_keys": sorted(only_r),
        "only_neu_keys": sorted(only_n),
        "shared_keys": sorted(shared),
        "only_real_flows": only_r_flows,
        "only_neu_flows": only_n_flows,
        "n_useful_only_real": n_useful_r,
        "n_useful_only_neu": n_useful_n,
        "ratio_useful": (n_useful_r / n_useful_n) if n_useful_n else (float("inf") if n_useful_r else None),
    }


def duplication_survival(
    dup_log: pd.DataFrame, registry: pd.DataFrame, freq_matrix: np.ndarray
) -> pd.DataFrame:
    """每次神经元重复事件: 复制出的边在最后一代的平均存在频率."""
    rows = []
    for _, r in dup_log.iterrows():
        gen, s, t = int(r["gen"]), int(r["src"]), int(r["dst"])
        copied = registry[
            (registry["added_gen"] == gen)
            & (registry["pre_idx"] == t)
            & (registry["origin"] == 1)
        ].index
        if len(copied) == 0:
            surv = np.nan
        else:
            surv = float(freq_matrix[-1, copied].mean())
        rows.append({"gen": gen, "src": int(s), "dst": t,
                     "n_copied": int(r["n_copied"]), "final_survival": surv})
    return pd.DataFrame(rows)


def _cell_name(nodes: pd.DataFrame, idx: int) -> str:
    r = nodes.iloc[idx]
    name = str(r["cell_type"]) if pd.notna(r["cell_type"]) else "unnamed"
    side = f"({r['side']})" if pd.notna(r["side"]) and str(r["side"]) != "center" else ""
    return f"{r['group']}/{name}{side}"


def _polarity_label(sign) -> str:
    """极性标签。接入递质口径后 sign 可以是 0 (调质类被排除, 见 nt_sign)。

    P1.4 之前只有 ±1 两种取值, 写 `'兴奋' if sign>0 else '抑制'` 是对的;
    现在 0 会被误标成「抑制」—— 而 DA/SER/OCT 在 `flygm`/`olfactory` 口径下
    恰恰是**不参与电流**的, 标成抑制会读成「它在压住下游」。
    """
    s = float(sign)
    if s > 0:
        return "兴奋"
    if s < 0:
        return "抑制"
    return "调质(不进和)"


def write_report(
    out, registry: pd.DataFrame, fix: dict, dup_surv: pd.DataFrame,
    struct_rows: list[dict], nodes: pd.DataFrame,
) -> None:
    if not struct_rows:
        return
    slog = pd.DataFrame(struct_rows)
    lines = [
        "# 结构进化报告",
        "",
        f"- 边宇宙: {int(slog.iloc[0]['universe'])} -> {int(slog.iloc[-1]['universe'])}"
        f" (累计新增 {int((slog['add'].sum()))} 条)",
        f"- 结构事件总计: 加边 {int(slog['add'].sum())}, 剪边 {int(slog['del'].sum())},"
        f" 神经元重复 {int(slog['dup'].sum())}",
        f"- 平均边保留率: {slog.iloc[-1]['mean_present']:.4f}"
        f" (初始 1.0000, 下降 = 结构分化)",
        "",
        f"## 固定的新连接 (频率≥{FIX_FREQ} 持续≥{SUSTAIN}代): **{fix['n_fixed']} 条**",
        f"- 早期漂移伪象 (已剔除): **{fix.get('n_drift_artifact', 0)} 条**",
        "",
    ]
    if fix["fixed_new_ids"]:
        rows = []
        for eid in fix["fixed_new_ids"]:
            r = registry.loc[eid]
            rows.append((float(r["final_freq"]), eid, r))
        rows.sort(reverse=True)
        lines += ["| 前元 | 后元 | 极性 | 加入代 | 末代频率 |", "|---|---|---|---|---|"]
        for ff, eid, r in rows[:15]:
            lines.append(
                f"| {_cell_name(nodes, int(r['pre_idx']))} "
                f"| {_cell_name(nodes, int(r['post_idx']))} "
                f"| {_polarity_label(r['sign'])} "
                f"| {int(r['added_gen'])} | {ff:.2f} |"
            )
    lines += [
        "",
        f"## 被剪除的真实连接 (频率≤{DEL_FREQ} 持续≥{SUSTAIN}代): **{fix['n_pruned']} 条**",
        "",
    ]
    if fix["pruned_real_ids"]:
        sample = fix["pruned_real_ids"][:10]
        lines += ["| 前元 | 后元 | 突触数 | 极性 |", "|---|---|---|---|"]
        for eid in sample:
            r = registry.loc[eid]
            lines.append(
                f"| {_cell_name(nodes, int(r['pre_idx']))} "
                f"| {_cell_name(nodes, int(r['post_idx']))} | - "
                f"| {_polarity_label(r['sign'])} |"
            )
    if len(dup_surv):
        good = dup_surv.dropna(subset=["final_survival"])
        lines += [
            "",
            "## 神经元重复谱系",
            "",
            f"- 重复事件 {len(dup_surv)} 次;"
            f" 复制边末代平均存续率 {good['final_survival'].mean():.3f}"
            if len(good) else "- 无可追踪复制边",
        ]
        top = good.sort_values("final_survival", ascending=False).head(5)
        for _, r in top.iterrows():
            lines.append(
                f"  - gen{int(r['gen'])}: {_cell_name(nodes, int(r['src']))} -> "
                f"{_cell_name(nodes, int(r['dst']))}, 复制 {int(r['n_copied'])} 条边,"
                f" 存续 {r['final_survival']:.2f}"
            )
    (out / "structure_report.md").write_text("\n".join(lines), encoding="utf-8")
