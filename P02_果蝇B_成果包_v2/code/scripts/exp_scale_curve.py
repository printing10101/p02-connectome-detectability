#!/usr/bin/env python
"""尺度插值实验: 在同一基质内扫描架构搜索空间规模, 画「架构红利 vs W」衰减曲线.

动机: 三基质定律(玩具35边有红利/连接组33K边归零/LLM开放空间归零)目前是三个
不同构基质的三个点, 规模变量与其他变量混杂. 本实验在玩具基质内部做尺度插值,
唯一变量 = 候选边数 W —— 把「三点点位」变成「剂量-响应曲线」.

构造: 12 个真节点(与 exp_g_transfer 完全同构, 参考线路/16 任务不变) + 诱饵节点
(固定偏置=0 不进基因组, 固定技能). 基因组: 架构层=W 条候选边掩码, 参数层=12 偏置.
跨规模公平性(唯一变的是覆盖难度):
  - 每基因组期望翻转数恒定 2.1 次/代 (p_flip = 2.1/W)
  - 初始随机边数恒定 9 条 (精确计数采样)
  - 参数空间恒定 11 维
条件: naive / param-only / directed(定向通道, 参考边集直装) × 3 种子 × 300 代.
预测: naive - param-only 随 W 单调衰减; directed 不随 W 变.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_g_transfer import make_reference, OPS, DOMAIN  # 任务与技能定义复用

OUT = Path(__file__).resolve().parents[1] / "results" / "exp_scale_curve"

SCALES = [(4, 4, 3), (12, 12, 9), (24, 24, 18), (36, 36, 27)]
CONDS = ("naive", "param-only", "directed")
SEEDS = (1, 2, 3)
GENS = 300
POP = 60
FLIPS_PER_GENOME = 2.1   # 每基因组期望架构翻转数, 跨规模恒定
INIT_EDGES = 9           # 初始随机边数(精确计数), 跨规模恒定
N_TRUE_BIAS = 11         # 真内节点 = S1(4) + S2(4) + S3(3); IN/OUT 无偏置


class Layout:
    """一个规模的基质: 真节点 id 固定(1-4/5-8/9-11/12), 诱饵节点按层追加."""

    def __init__(self, w1: int, w2: int, w3: int):
        self.w = (w1, w2, w3)
        self.s1 = list(range(1, 5)) + list(range(13, 13 + (w1 - 4)))
        self.s2 = list(range(5, 9)) + list(range(13 + (w1 - 4), 13 + (w1 - 4) + (w2 - 4)))
        self.s3 = list(range(9, 12)) + list(range(13 + (w1 - 4) + (w2 - 4),
                                                  13 + (w1 - 4) + (w2 - 4) + (w3 - 3)))
        self.out = 12
        self.n_nodes = 1 + len(self.s1) + len(self.s2) + len(self.s3) + 1
        self.true_internal = list(range(1, 12))          # 11 个真内节点(不含 IN/OUT)
        e_in = [(0, s) for s in self.s1]
        e_12 = [(a, b) for a in self.s1 for b in self.s2]
        e_23 = [(a, b) for a in self.s2 for b in self.s3]
        e_3o = [(s, self.out) for s in self.s3]
        self.edges = e_in + e_12 + e_23 + e_3o
        self.W = len(self.edges)
        self.eindex = {p: i for i, p in enumerate(self.edges)}
        # 参考边集: 与 exp_g_transfer 相同的 14 条真-真边
        ref_pairs = [(0, 1), (0, 2), (0, 3), (1, 5), (2, 6), (3, 7), (4, 8),
                     (1, 6), (3, 5), (5, 9), (6, 10), (7, 11), (8, 11), (9, 12)]
        self.ref_mask = np.zeros(self.W, dtype=bool)
        for p in ref_pairs:
            self.ref_mask[self.eindex[p]] = True
        # 按层预分组边; 层成员用阶段列表而非 id 区间——
        # 真 S2/S3 的 id 数值上落在真 S1 之后, 用区间会把高层节点塞进低层求值(W>35 全错)
        self.layer_edges = [
            [(self.eindex[p], p[0], p[1]) for p in pairs] for pairs in (e_in, e_12, e_23, e_3o)
        ]
        self.layer_nodes = [self.s1, self.s2, self.s3, [self.out]]

    def random_init(self, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
        mask = np.zeros(self.W, dtype=bool)
        for i in rng.sample(range(self.W), INIT_EDGES):
            mask[i] = True
        return mask, np.zeros(N_TRUE_BIAS)

    def full_biases(self, bias12: np.ndarray) -> np.ndarray:
        full = np.zeros(self.n_nodes)
        full[self.true_internal] = bias12
        return full

    def run(self, mask: np.ndarray, bias12: np.ndarray, xs: np.ndarray) -> np.ndarray:
        full = self.full_biases(bias12)
        vals = np.zeros((self.n_nodes, len(xs)), dtype=np.int64)
        vals[0] = xs.astype(np.int64)
        for edges, nodes in zip(self.layer_edges, self.layer_nodes):
            for node in nodes:
                acc = np.zeros(len(xs), dtype=np.int64)
                for ei, src, dst in edges:
                    if dst == node and mask[ei]:
                        acc += vals[src]
                vals[node] = OPS[(node - 1) % 4](np.round(acc + full[node]) % DOMAIN)
        return vals[self.out].astype(float)

    def solved(self, mask, bias12, xs, ys) -> int:
        return int((np.abs(self.run(mask, bias12, xs) - ys) <= 0.5).sum())


def evolve(layout: Layout, cond: str, seed: int, xs, ys, gens: int) -> dict:
    rng = random.Random(seed * 131 + 7)
    nprng = np.random.default_rng(seed)
    mask0 = np.zeros(layout.W, dtype=bool)
    bias0 = np.zeros(N_TRUE_BIAS)
    if cond == "directed":
        mask0 = layout.ref_mask.copy()
    else:
        mask0, bias0 = layout.random_init(rng)

    def score(m, b):
        return layout.solved(m, b, xs, ys)

    pop = [(mask0.copy(), bias0.copy()) for _ in range(POP)]
    if cond != "directed":
        pop = [layout.random_init(rng) for _ in range(POP)]
    fit = np.array([score(m, b) for m, b in pop], dtype=float)
    p_flip = FLIPS_PER_GENOME / layout.W

    best_hist, last_improve = [], 0
    for gen in range(gens):
        arch_ok = cond == "naive"
        order = np.argsort(-fit)
        new_pop = [pop[order[0]], pop[order[1]]]
        while len(new_pop) < POP:
            a, b = rng.randrange(POP), rng.randrange(POP)
            if fit[a] < fit[b]:
                a = b
            ma, ba = pop[a]
            c, d = rng.randrange(POP), rng.randrange(POP)
            if fit[c] < fit[d]:
                c = d
            mc, bc = pop[c]
            cm = np.where(nprng.random(layout.W) < 0.5, ma, mc)
            cb = np.where(nprng.random(N_TRUE_BIAS) < 0.5, ba, bc)
            if arch_ok and rng.random() < 0.7:      # 架构突变(期望 2.1 翻转)
                flips = nprng.random(layout.W) < p_flip
                cm = cm ^ flips
            if nprng.random() < 0.3:                # 参数突变
                cb = cb + nprng.normal(0, 1.2, N_TRUE_BIAS) * (nprng.random(N_TRUE_BIAS) < 0.5)
            new_pop.append((cm, cb))
        pop = new_pop
        fit = np.array([score(m, b) for m, b in pop], dtype=float)
        best = float(fit.max())
        if best > (max(best_hist) if best_hist else -1.0):
            last_improve = gen
        best_hist.append(best)

    order = np.argsort(-fit)
    return {"cond": cond, "seed": seed, "final_best": float(fit.max()),
            "val_final": float(fit.mean()), "W": layout.W,
            "final_mask_on": int(pop[order[0]][0].sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟: 2 规模 x 1 种子 x 40 代")
    args = ap.parse_args()
    scales, seeds, gens = ((SCALES[:2], (1,), 40) if args.quick else (SCALES, SEEDS, GENS))

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref_mask, ref_bias, xs, ys = make_reference()   # 任务与参考线路与实验 G 完全一致

    results = []
    for w in scales:
        layout = Layout(*w)
        # 布局自检: 小规模必须与 exp_g_transfer 的 35 边宇宙完全一致
        if w == (4, 4, 3):
            base = __import__("exp_g_transfer")
            assert layout.W == base.N_EDGES and layout.n_nodes == base.N_NODES
        row = {"w1w2w3": w, "W": layout.W, "conds": {}}
        for cond in CONDS:
            finals = []
            for seed in seeds:
                r = evolve(layout, cond, seed, xs, ys, gens=gens)
                finals.append(r["final_best"])
            row["conds"][cond] = {"mean": float(np.mean(finals)),
                                  "per_seed": [float(f) for f in finals]}
            print(f"W={layout.W:<5} {cond:<11} mean={row['conds'][cond]['mean']:.2f} "
                  f"seeds={row['conds'][cond]['per_seed']}", flush=True)
        row["arch_value"] = row["conds"]["naive"]["mean"] - row["conds"]["param-only"]["mean"]
        results.append(row)
        print(f"== W={layout.W}: 架构红利 = {row['arch_value']:.2f} ==", flush=True)

    # ---- 判定 ----
    av = [r["arch_value"] for r in results]
    ws = [r["W"] for r in results]
    decay_steps = sum(1 for i in range(1, len(av)) if av[i] <= av[i - 1])
    ratio = av[-1] / av[0] if av[0] > 0 else None
    if decay_steps == len(av) - 1 and (ratio is not None and ratio < 1 / 3):
        verdict = "DECAY_CONFIRMED"
    elif decay_steps >= len(av) - 2:
        verdict = "DECAY_PARTIAL"
    else:
        verdict = "DECAY_NOT_OBSERVED"

    summary = {
        "protocol": (
            "同基质尺度插值: 12 真节点 + 诱饵节点(冻结偏置); 16 任务不变; "
            f"每基因组期望翻转 2.1/代(跨规模恒定), 初始随机边 9 条(恒定), 参数空间恒定 12 维; "
            f"{len(CONDS)}条件 x {len(seeds)}种子 x {gens}代"
        ),
        "scales": results,
        "arch_value_by_W": dict(zip(ws, [round(a, 2) for a in av])),
        "decay_steps": f"{decay_steps}/{len(av) - 1}",
        "largest_over_smallest": round(ratio, 3) if ratio is not None else None,
        "verdict": verdict,
        "verdict_rule": "CONFIRMED: 逐点不增 且 末点<首点1/3; PARTIAL: 至少逐点不增减一步; 其余 NOT_OBSERVED",
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    (OUT / "scale_verdict.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 曲线图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cond, label in (("naive", "naive(通用双层)"), ("param-only", "param-only(仅参数)"),
                            ("directed", "directed(定向通道)")):
            ys_ = [r["conds"][cond]["mean"] for r in results]
            ax.plot(ws, ys_, marker="o", label=label)
        ax.set_xscale("log")
        ax.set_xlabel("候选边数 W (log)")
        ax.set_ylabel("终局解题数 /16 (3 种子均值)")
        ax.set_title("架构红利随搜索空间规模衰减")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "scale_curve.png", dpi=130)
        summary["plot"] = "scale_curve.png"
    except Exception as e:
        print("plot skip:", e)

    (OUT / "scale_verdict.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("arch_value_by_W", "decay_steps", "largest_over_smallest", "verdict", "wall_min")},
                     ensure_ascii=False, indent=1))
    print("WROTE", OUT / "scale_verdict.json")


if __name__ == "__main__":
    main()
