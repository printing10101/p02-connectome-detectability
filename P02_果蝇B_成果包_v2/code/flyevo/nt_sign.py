"""递质符号策略 (Eckstein 2024 递质表接入; 任务 P1.4).

背景
----
v1 的符号规则是「GABA 抑制, 其余一律兴奋」(见 build_subcircuit.py)。这条规则把
GLUT / DA / SER / OCT 全部当成兴奋 —— 在果蝇中央脑这是错的, 而本项目的子回路
(ORN→PN→LH→DN) 正是嗅觉通路, 恰好是这条错误最要命的地方:

- **GLUT 在果蝇嗅觉系统里是抑制性的** (Liu & Wilson, PNAS 2013,
  "Glutamate is an inhibitory neurotransmitter in the Drosophila olfactory system")。
  本项目子回路里 GLUT 边占 9.6% —— 按 v1 规则它们全被当成兴奋。
- **DA / SER / OCT 是调质性的**, 既不是快速兴奋也不是快速抑制。FlyGM 的
  W = N_exc − N_inh (FlyGM eq.1) 只对兴奋/抑制两类求和, 调质突触因此**不进和**。

两套公开口径不一致, 所以本模块不替研究者做选择, 而是把三条口径都实现出来:

| policy       | 兴奋集            | 抑制集           | 来源 |
|--------------|-------------------|------------------|------|
| `gaba-only`  | 除 GABA 外全部    | GABA             | v1 基线 (本项目历史口径) |
| `flygm`      | ACH, GLU, ASP, HIS| GABA, GLY        | FlyGM 论文原文符号表 |
| `olfactory`  | ACH, ASP, HIS     | GABA, GLY, GLU   | 嗅觉系统文献 (GLU 抑制) |
| `none`       | 全部              | —                | 无符号对照 |

三者的差别只有一件事: **GLU 算兴奋还是抑制**。这正是本项目尺度实验 (exp_scale_curve_v2)
里「值钱的架构改动存在但选择看不见」所指的那类改动 —— 所以符号口径本身就是一个
可检验的变量, 不该被默认值藏起来。

W = N_exc − N_inh 的落地方式
----------------------------
FlyWire 的 connections 表按 (pre, post, neuropil) 给行, 每行一个 `nt_type`。
本项目的边表就是这些行 (33,527 行 = 22,403 个神经元对)。于是一条边的贡献是

    w_ij * sign(nt_type) * syn_count

对一个神经元对 (i, j) 的所有行求和, 得到

    Σ_rows sign_row × syn_count_row = N_exc(i,j) − N_inh(i,j)

即 **FlyGM eq.1 的净极化突触计数, 在神经元对层面精确成立** —— 不需要额外的
per-synapse 表 (那需要 ~50 GB 的 synapses.csv)。子回路里 1,151 个神经元对 (5.1%)
跨 neuropil 出现混合递质, 求和会自然处理, 不需要特判。

用法
----
    from flyevo.nt_sign import POLICIES, load_substrate
    sign, w0, meta = load_substrate(policy="flygm", weight_mode="syn")

`weight_mode`:
  - `unit`     w0 ≡ 1        (v1 行为, 逐位可复现)
  - `syn`      w0 = syn_count / mean(syn_count)
  - `sqrt-syn` w0 = sqrt(syn_count / mean(syn_count))   (压掉长尾, 保守变体)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# FlyWire 的 nt_type 取值是 ACH / GABA / GLUT / DA / SER / OCT;
# 文献里常写 GLU / ACh / 5-HT。统一到 FlyWire 口径, 别名在 canon() 里折叠 ——
# 第一版忘了这一步, 导致 "GLUT" 既不匹配兴奋集也不匹配抑制集, 被当成调质剔除,
# flygm 与 olfactory 两条口径静默退化成同一个。
NT_ALIASES = {
    "GLU": "GLUT", "GLUTAMATE": "GLUT",
    "ACH": "ACH", "ACETYLCHOLINE": "ACH",
    "GABA": "GABA",
    "GLY": "GLY", "GLYCINE": "GLY",
    "ASP": "ASP", "ASPARTATE": "ASP",
    "HIS": "HIS", "HISTAMINE": "HIS",
    "DA": "DA", "DOPAMINE": "DA",
    "SER": "SER", "5-HT": "SER", "SEROTONIN": "SER",
    "OCT": "OCT", "OCTOPAMINE": "OCT",
}


def canon(nt: str) -> str:
    return NT_ALIASES.get(str(nt).strip().upper(), str(nt).strip().upper())


EXC_FLYGM = frozenset({"ACH", "GLUT", "ASP", "HIS"})
INH_FLYGM = frozenset({"GABA", "GLY"})
EXC_OLFACTORY = frozenset({"ACH", "ASP", "HIS"})
INH_OLFACTORY = frozenset({"GABA", "GLY", "GLUT"})
MODULATORY = frozenset({"DA", "SER", "OCT"})

POLICIES: dict[str, dict] = {
    # v1 基线: 只有 GABA 抑制, 其余(含 GLU/DA/SER/OCT)一律兴奋
    "gaba-only": {"exc": None, "inh": frozenset({"GABA"}), "mod": "exc",
                  "source": "v1 baseline (本项目历史口径)"},
    # FlyGM 论文原文符号表: GLU 兴奋; DA/SER/OCT 未分类 -> 不进和
    "flygm": {"exc": EXC_FLYGM, "inh": INH_FLYGM, "mod": "exclude",
              "source": "FlyGM (arXiv 2602.17997) 符号表; W = N_exc - N_inh"},
    # 嗅觉系统文献口径: GLU 抑制
    "olfactory": {"exc": EXC_OLFACTORY, "inh": INH_OLFACTORY, "mod": "exclude",
                  "source": "Liu & Wilson PNAS 2013: 果蝇嗅觉系统 GLU 抑制"},
    # 无符号对照: 全兴奋
    "none": {"exc": None, "inh": frozenset(), "mod": "exc",
             "source": "无符号对照 (signs vs no-signs sanity)"},
}

WEIGHT_MODES = ("unit", "syn", "sqrt-syn")
NT_TABLE = DATA / "flywire_subcircuit_nt.csv"
EDGE_TABLE = DATA / "flywire_subcircuit_edges.csv"


def sign_of_nt(nt: str, policy: str) -> float:
    """单条边的符号。`gaba-only` 与 `none` 的 exc=None 表示「兜底为兴奋」。"""
    p = POLICIES[policy]
    c = canon(nt)
    if c in p["inh"]:
        return -1.0
    if p["exc"] is not None and c not in p["exc"]:
        # 既不在兴奋集也不在抑制集 -> 调质
        return 0.0 if p["mod"] == "exclude" else 1.0
    return 1.0


def sign_array(nt_types, policy: str) -> np.ndarray:
    lut = {canon(nt): sign_of_nt(nt, policy) for nt in set(map(str, nt_types))}
    return np.array([lut[canon(t)] for t in nt_types], dtype=np.float64)


def base_weight(syn_counts, mode: str = "unit") -> np.ndarray:
    s = np.asarray(syn_counts, dtype=np.float64)
    if mode == "unit":
        return np.ones_like(s)
    m = float(s.mean()) or 1.0
    if mode == "syn":
        return s / m
    if mode == "sqrt-syn":
        return np.sqrt(np.maximum(s, 0.0) / m)
    raise ValueError(f"未知 weight_mode: {mode}")


def net_polarized(nt_types, syn_counts, policy: str) -> np.ndarray:
    """逐边的带符号突触计数 sign(nt) * syn_count。按神经元对求和 = N_exc − N_inh。"""
    return sign_array(nt_types, policy) * np.asarray(syn_counts, dtype=np.float64)


def load_substrate(
    policy: str = "gaba-only",
    weight_mode: str = "unit",
    edge_table: Path | None = None,
    nt_table: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回 (sign, w0, meta)。默认 (gaba-only, unit) 与 v1 行为逐位一致。

    优先读 `flywire_subcircuit_nt.csv` (由 scripts/build_nt_table.py 生成);
    若不存在则回落到边表自带的 `sign` 列 + 单位权重。
    """
    et = Path(edge_table) if edge_table else EDGE_TABLE
    nt = Path(nt_table) if nt_table else NT_TABLE
    edges = pd.read_csv(et)

    if not nt.exists():
        if policy == "gaba-only" and weight_mode == "unit":
            return (edges["sign"].to_numpy(dtype=np.float64),
                    np.ones(len(edges)),
                    {"source": "edges.sign (v1 fallback)", "policy": policy,
                     "weight_mode": weight_mode, "n_edges": len(edges)})
        raise FileNotFoundError(
            f"{nt} 不存在 —— 先跑 scripts/build_nt_table.py 才能用 policy={policy}"
        )

    t = pd.read_csv(nt)
    if len(t) != len(edges):
        raise RuntimeError(
            f"递质表 {len(t)} 行 != 边表 {len(edges)} 行 —— 两表必须逐行对齐"
        )
    col = f"sign_{policy.replace('-', '_')}"
    if col not in t.columns:
        raise ValueError(f"递质表缺列 {col}; 现有 {list(t.columns)}")
    sign = t[col].to_numpy(dtype=np.float64)
    w0 = base_weight(t["syn_count"].to_numpy(), weight_mode)
    meta = {
        "source": str(nt.name), "policy": policy, "weight_mode": weight_mode,
        "n_edges": int(len(t)),
        "n_sign_pos": int((sign > 0).sum()),
        "n_sign_neg": int((sign < 0).sum()),
        "n_sign_zero": int((sign == 0).sum()),
        "policy_source": POLICIES[policy]["source"],
        "mean_w0": float(w0.mean()),
    }
    return sign, w0, meta


def summary() -> dict:
    """递质表口径概览 (给报告用)。"""
    out: dict = {"policies": {k: {**v, "exc": sorted(v["exc"]) if v["exc"] else None,
                                  "inh": sorted(v["inh"])} for k, v in POLICIES.items()},
                 "weight_modes": list(WEIGHT_MODES),
                 "modulatory_classes": sorted(MODULATORY)}
    if NT_TABLE.exists():
        t = pd.read_csv(NT_TABLE)
        out["table"] = {
            "path": str(NT_TABLE.name), "n_edges": int(len(t)),
            "syn_total": int(t["syn_count"].sum()),
            "nt_composition": t["nt_type"].value_counts().to_dict(),
            "sign_counts": {k: {"pos": int((t[f"sign_{k.replace('-', '_')}"] > 0).sum()),
                                "neg": int((t[f"sign_{k.replace('-', '_')}"] < 0).sum()),
                                "zero": int((t[f"sign_{k.replace('-', '_')}"] == 0).sum())}
                            for k in POLICIES},
            "pairs_with_mixed_nt": int(t["pair_mixed"].sum()) if "pair_mixed" in t else None,
        }
    return out


if __name__ == "__main__":
    print(json.dumps(summary(), ensure_ascii=False, indent=2))
