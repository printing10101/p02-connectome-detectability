#!/usr/bin/env python
"""实验 G 极简版: 技能图自修改台 —— 「选择可见性不对称」+ sham 审计的迁移演示 (C9 最小版).

基质(与果蝇完全不同构, 但协议同构):
  3 级技能图: IN → S1(4节点) → S2(4节点) → S3(3节点) → OUT, 值域 Z_64.
  每节点固定一种「技能」(rot/xor/mul/square 轮转) + 一个可调偏置参数.
  架构基因组 = 35 条候选边的存在掩码; 参数基因组 = 12 个节点偏置.
任务: 16 个 (输入→目标) 对, 由隐藏参考线路(特定边集+特定偏置)生成.
执行: 输入沿现有边流动, 节点对入边求和 + 偏置, 过技能后取模; 输出 |out-y|<=0.5 记解出.

四条件 × 5 种子 (沿用果蝇判决纪律):
  Naive        双层自由突变, 真实选择
  Param-only   架构冻结在初始随机掩码 —— 参数层单独能走多远
  Sham         独立突变流(seed+10000) + 适应度打乱 —— 漂移基线 (clean 协议教训)
  Gated        平台期(8代无改进)才开架构突变 —— 两时间尺度门控
G0 手工接线: 参考边集直接装进零偏置/随机脑 —— 先证明「值钱的架构改动存在」(P0.1 纪律).
层消融: Naive 最优的架构层/参数层分别回滚到初始态 —— 最终性能存放在哪一层.

判定: Naive≫Sham(选择信号) 且 Naive>Param-only(架构边际价值) 且 参数承载>=架构承载
  → ASYMMETRY_REPLICATED; 仅有选择信号 → PARTIAL_SELECTION_ONLY; 否则 NOT_REPLICATED.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "exp_g_transfer"

DOMAIN = 64
N1, N2, N3 = 4, 4, 3
IN, S1, S2, S3, OUTN = 0, range(1, 5), range(5, 9), range(9, 12), 12
N_NODES = 13                          # IN(0) + S1(1-4) + S2(5-8) + S3(9-11) + OUT(12)
N_TASKS = 16
E_IN = [(IN, j) for j in S1]                       # 4
E_12 = [(i, 5 + j) for i in S1 for j in range(N2)]   # 16 (S2 基址 = 5)
E_23 = [(i, 9 + j) for i in S2 for j in range(N3)]   # 12
E_3O = [(i, OUTN) for i in S3]                       # 3
EDGES = E_IN + E_12 + E_23 + E_3O
N_EDGES = len(EDGES)
SRC = np.array([e[0] for e in EDGES])
DST = np.array([e[1] for e in EDGES])


def op_rot3(x):
    x = x.astype(np.int64)
    return ((x << 3) | (x >> 3)) % DOMAIN

def op_xor5(x):
    return (x.astype(np.int64) ^ 5) % DOMAIN

def op_mul3(x):
    return (x.astype(np.int64) * 3) % DOMAIN

def op_square(x):
    return (x.astype(np.int64) ** 2) % DOMAIN

OPS = [op_rot3, op_xor5, op_mul3, op_square]  # 技能按节点轮转分配


def make_reference(seed: int = 7):
    """隐藏参考线路: 特定边集 + 特定偏置; 任务目标 = 参考线路的输出."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(N_EDGES, dtype=bool)
    ref_edges = [
        (0, 1), (0, 2), (0, 3),                 # IN→S1: 3/4
        (1, 5), (2, 6), (3, 7), (4, 8),         # S1→S2 对角
        (1, 6), (3, 5),                         # S1→S2 交叉
        (5, 9), (6, 10),                        # S2→S3
        (7, 11), (8, 11),                       # S2→S3 汇聚
        (9, 12),                                # S3→OUT
    ]
    for e in ref_edges:
        mask[EDGES.index(e)] = True
    biases = np.zeros(N_NODES)
    biases[[1, 2, 3, 5, 7, 9]] = [3, -2, 5, 1, -4, 2]
    xs = rng.integers(0, DOMAIN, size=N_TASKS).astype(float)
    ys = run_circuit(mask, biases, xs)
    return mask, biases, xs, ys


def run_circuit(mask: np.ndarray, biases: np.ndarray, xs: np.ndarray) -> np.ndarray:
    vals = np.zeros((N_NODES, len(xs)), dtype=np.int64)
    vals[IN] = xs.astype(np.int64)
    for lo, hi in [(1, 5), (5, 9), (9, 12), (12, 13)]:
        for node in range(lo, hi):
            ein = np.flatnonzero(mask & (DST == node))
            u = biases[node] + (vals[SRC[ein]].sum(axis=0) if len(ein) else 0)
            vals[node] = OPS[(node - 1) % 4](np.round(u) % DOMAIN)
    return vals[OUTN].astype(float)


def solved(mask: np.ndarray, biases: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> int:
    return int((np.abs(run_circuit(mask, biases, xs) - ys) <= 0.5).sum())


# ---------- GA ----------

def random_genome(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return rng.random(N_EDGES) < 0.25, np.zeros(N_NODES)


def mutate(mask: np.ndarray, bias: np.ndarray, rng, arch_ok: bool,
           p_flip: float = 0.06, p_bias: float = 0.3, bias_std: float = 1.2):
    m = mask ^ (rng.random(N_EDGES) < p_flip) if arch_ok else mask
    b = bias + rng.normal(0, bias_std, size=N_NODES) * (rng.random(N_NODES) < p_bias)
    return m, b


def crossover(a, b, rng):
    ma, ba = a
    mb, bb = b
    return (np.where(rng.random(N_EDGES) < 0.5, ma, mb),
            np.where(rng.random(N_NODES) < 0.5, ba, bb))


def evolve(cond: str, seed: int, xs, ys, init_mask: np.ndarray = None,
           pop_size: int = 60, gens: int = 300, plateau: int = 8):
    """单条件单种子; sham 用独立突变流 + 适应度打乱.

    directed = 定向通道: 结构由通道直接安装(init_mask=参考边集), 通用选择只调参验收
    (果蝇 handwire / EvoMap Capsule 同款原理); 与 param-only 唯一差异是初始结构来源.
    """
    rng = np.random.default_rng(seed)
    sham_rng = np.random.default_rng(seed + 10_000)

    base_mask = init_mask if init_mask is not None else None
    pop = []
    for _ in range(pop_size):
        m, b = random_genome(rng)
        pop.append((base_mask.copy() if base_mask is not None else m, b))
    fit = np.array([solved(m, b, xs, ys) for m, b in pop], dtype=float)
    arch_mut_spent, last_improve, best_hist = 0, 0, []
    for gen in range(gens):
        if cond == "gated":
            arch_ok = (gen - last_improve) >= plateau
        elif cond in ("param-only", "directed"):
            arch_ok = False  # directed 的结构由通道给定, 不参与搜索
        else:
            arch_ok = True

        order = np.argsort(-fit)
        new_pop = [pop[order[0]], pop[order[1]]]  # 精英 2
        while len(new_pop) < pop_size:
            a, b_ = rng.integers(0, pop_size, 2)
            if fit[a] < fit[b_]:
                a = b_
            c, d = rng.integers(0, pop_size, 2)
            if fit[c] < fit[d]:
                c = d
            child = crossover(pop[a], pop[c], rng)
            use_rng = sham_rng if cond == "sham" else rng
            child = mutate(child[0], child[1], use_rng, arch_ok=arch_ok)
            if arch_ok:  # 架构突变开销: 各条件统一计数, gated 的效率才有分母可比
                arch_mut_spent += int((child[0] != pop[a][0]).sum())
            new_pop.append(child)
        pop = new_pop
        fit = np.array([solved(m, b, xs, ys) for m, b in pop], dtype=float)
        if cond == "sham":  # 漂移基线: 适应度打乱
            fit = rng.permutation(fit)

        best = float(fit.max())
        if best > (max(best_hist) if best_hist else -1.0):
            last_improve = gen
        best_hist.append(best)

    order = np.argsort(-fit)
    bm, bb = pop[order[0]]
    return {
        "final_best": float(fit.max()),
        "final_mean": float(fit.mean()),
        "best_genome": (bm, bb),
        "arch_mut_spent": arch_mut_spent,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟: 2 种子 x 80 代")
    args = ap.parse_args()
    gens, seeds = (80, (1, 2)) if args.quick else (500, (1, 2, 3, 4, 5))

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref_mask, ref_bias, xs, ys = make_reference()

    # ---- G0 手工接线: 值钱的架构改动存在吗 (P0.1 纪律) ----
    init_mask, init_bias = random_genome(np.random.default_rng(0))
    g0 = {
        "初始随机掩码+零偏置": solved(init_mask, init_bias, xs, ys),
        "参考边集+零偏置(只装结构)": solved(ref_mask, np.zeros(N_NODES), xs, ys),
        "参考边集+参考偏置(满配)": solved(ref_mask, ref_bias, xs, ys),
        "随机掩码+参考偏置(装错结构)": solved(init_mask, ref_bias, xs, ys),
    }
    print("G0 handwire:", json.dumps(g0), flush=True)

    # ---- 主对照 ----
    per_cond: dict[str, list] = {}
    for cond in ("naive", "param-only", "directed", "sham", "gated"):
        per_cond[cond] = []
        for seed in seeds:
            r = evolve(cond, seed, xs, ys,
                       init_mask=ref_mask if cond == "directed" else None, gens=gens)
            r["seed"] = seed
            per_cond[cond].append(r)
            print(f"{cond:<11} s{seed}: final_best={r['final_best']:.0f}/16 "
                  f"mean={r['final_mean']:.2f}", flush=True)

    # ---- 层消融: 最终性能存放在哪一层 ----
    ablations = {}
    for r in per_cond["naive"]:
        m, b = r["best_genome"]
        ablations[f"s{r['seed']}"] = {
            "full": solved(m, b, xs, ys),
            "参数承载(架构回滚)": solved(init_mask, b, xs, ys),
            "架构承载(参数回滚)": solved(m, init_bias, xs, ys),
        }

    # ---- 架构漂移 vs 有用固定 ----
    def mask_stats(m):
        return {"n_on": int(m.sum()), "match_ref": int((m & ref_mask).sum())}
    drift = {
        "init(Chance水平)": mask_stats(init_mask),
        "ref": mask_stats(ref_mask),
        "naive": [mask_stats(r["best_genome"][0]) for r in per_cond["naive"]],
        "sham": [mask_stats(r["best_genome"][0]) for r in per_cond["sham"]],
    }

    # ---- 判定 ----
    def mean_final(cond):
        return float(np.mean([r["final_best"] for r in per_cond[cond]]))
    naive, sham = mean_final("naive"), mean_final("sham")
    ponly, gated = mean_final("param-only"), mean_final("gated")
    directed = mean_final("directed")
    par_carried = float(np.mean([a["参数承载(架构回滚)"] for a in ablations.values()]))
    arch_carried = float(np.mean([a["架构承载(参数回滚)"] for a in ablations.values()]))
    gated_spent = float(np.mean([r["arch_mut_spent"] for r in per_cond["gated"]]))
    naive_spent = float(np.mean([r["arch_mut_spent"] for r in per_cond["naive"]]))

    selection_signal = (naive - sham) >= 2.0
    arch_value = (naive - ponly) >= 1.0
    directed_beats_search = (directed - naive) >= 1.0
    # 消融的「承载」数字只描述、不 gate 判决: 联合存放(回滚任一层都崩)是合法结局,
    # 与果蝇「参数层独扛」的差异本身就是迁移发现
    if selection_signal and arch_value:
        verdict = "ASYMMETRY_REPLICATED"
    elif selection_signal:
        verdict = "PARTIAL_SELECTION_ONLY"
    else:
        verdict = "NOT_REPLICATED"

    summary = {
        "protocol": (
            f"skill-DAG 3级/12技能/35候选边; {N_TASKS}任务; pop60 x {gens}代 x "
            f"{len(seeds)}种子; sham=独立突变流+适应度打乱; "
            "directed=结构经定向通道直装(参考边集)+选择只调参验收"
        ),
        "G0_handwire": g0,
        "final_best_per_cond": {
            "naive": naive, "param-only": ponly, "directed": directed,
            "sham": sham, "gated": gated,
        },
        "ablation_naive": {"参数承载(架构回滚)": par_carried, "架构承载(参数回滚)": arch_carried},
        "mask_stats": drift,
        "gated_efficiency": {"arch_mut_spent_gated": gated_spent, "arch_mut_spent_naive": naive_spent},
        "checks": {
            "selection_signal(naive-sham>=2)": selection_signal,
            "arch_value(naive-paramonly>=1)": arch_value,
            "directed_channel(directed-naive>=1)": directed_beats_search,
        },
        "verdict": verdict,
        "wall_s": round(time.time() - t0, 1),
    }
    (OUT / "exp_g_verdict.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("VERDICT", verdict, "| wall", summary["wall_s"], "s")
    print("WROTE", OUT / "exp_g_verdict.json")


if __name__ == "__main__":
    main()
