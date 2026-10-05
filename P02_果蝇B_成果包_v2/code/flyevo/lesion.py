"""桥1 L5: 切断主感觉通路, 使权重无法再解决任务, 逼结构加边绕行.

对应 GlossoGen「预算收到极紧」的硬件版: 现有线路物理上传不过去.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# group 名与 build_subcircuit 一致
GROUP_ORN = "ORN"
GROUP_PN = "PN"
GROUP_LH = "LH"
GROUP_DN = "DN"


@dataclass
class LesionSpec:
    frac: float = 0.7          # 切断比例
    pathway: str = "ORN_PN"    # ORN_PN | PN_LH | ORN_PN_PN_LH
    seed: int = 0
    keep_bypass: bool = True   # 保留少量直达旁路, 让「绕行」可被选择


def _group_idx(nodes, name: str) -> np.ndarray:
    return np.asarray(nodes.index[nodes["group"] == name].to_numpy(), dtype=np.int64)


def lesion_mask(
    pre: np.ndarray,
    post: np.ndarray,
    nodes,
    spec: LesionSpec,
) -> np.ndarray:
    """返回与边等长的 bool 掩码; False = 该边被切断(权重评估为 0).

    nodes: 需含列 group, 索引 = 神经元全局 id (与 edges 的 pre/post 一致)
    """
    n = len(pre)
    keep = np.ones(n, dtype=bool)
    rng = np.random.default_rng(spec.seed)

    orn = _group_idx(nodes, GROUP_ORN)
    pn = _group_idx(nodes, GROUP_PN)
    lh = _group_idx(nodes, GROUP_LH)

    orn_set = set(orn.tolist())
    pn_set = set(pn.tolist())
    lh_set = set(lh.tolist())

    cut = np.zeros(n, dtype=bool)
    for i in range(n):
        a, b = int(pre[i]), int(post[i])
        on_orn_pn = a in orn_set and b in pn_set
        on_pn_lh = a in pn_set and b in lh_set
        if spec.pathway == "ORN_PN" and on_orn_pn:
            cut[i] = True
        elif spec.pathway == "PN_LH" and on_pn_lh:
            cut[i] = True
        elif spec.pathway == "ORN_PN_PN_LH" and (on_orn_pn or on_pn_lh):
            cut[i] = True

    idx = np.flatnonzero(cut)
    if len(idx) == 0:
        return keep

    n_cut = int(round(spec.frac * len(idx)))
    # 可选: 保证切断的是随机子集, 避免总是同侧
    chosen = rng.choice(idx, size=n_cut, replace=False) if n_cut > 0 else np.array([], dtype=int)
    keep[chosen] = False

    # 旁路保护: 保留 ORN→LH / PN→DN 等非主通路 (本就未 cut)
    if not spec.keep_bypass:
        # 额外也切断 ORN→LH 直达, 逼更极端的结构搜索
        for i in range(n):
            if int(pre[i]) in orn_set and int(post[i]) in lh_set:
                keep[i] = False

    return keep


def apply_lesion_to_weights(weights: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """把切断掩码乘到权重向量上."""
    return weights * keep.astype(weights.dtype)


def lesion_summary(keep: np.ndarray) -> dict:
    n = len(keep)
    n_cut = int((~keep).sum())
    return {
        "n_edges": n,
        "n_cut": n_cut,
        "cut_frac": n_cut / n if n else 0.0,
    }
