#!/usr/bin/env python
"""世代数 x 变异率 x 规模 三因子交互扫描 (任务 P1.6).

要解释的异常
------------
v1 尺度插值 (300 代) 在 W=35 处得到 **param-only (10.0) > naive (7.67)** —— 小空间里
架构突变是**负资产**。但实验 G (500 代, 同构基质 W=35) 的方向相反: Naive 9.2 >
Param-only 6.2。同一基质、同一规模, 两个方向。

假设: 架构突变 = 发现 − 破坏, 两项时间尺度不同
-----------------------------------------------
- **破坏即时**。p_flip = F/W 之下, 每代被翻掉的真边期望数 = F x (14/W)。W=35, F=2.1
  时 = 0.84 条/代 —— 参数层刚调好的脚手架每代被拆掉将近一条边。
- **发现稀有**。要把该补的那一条边从 W 个候选里翻出来, 时间随 W 增长; 而且翻出来之后
  还要参数层配合才能兑现成分数。

净收益 = 发现(慢, 随代际累积) − 破坏(即时, 恒定) → 必然存在**交叉世代数**: 之前
param-only 领先, 之后 naive-joint 反超。交叉点应随 W 增大而后移, 随变异率增大而后移。

三个条件 (同起点: 参考边集 + 零偏置)
------------------------------------
  naive-joint       架构自由 + 参数自由   -> 主曲线
  param-only        架构冻结 + 参数自由   -> 反事实 (无架构突变的收益)
  naive-arch-pinned 架构自由 + 参数钉在参考偏置 -> **纯破坏读数**
                    起点就是 16/16 满解, 曲线只能往下走 —— 直接量化「更大的变异邻域
                    对已知好设计有多有害」, 且不受参数层追赶的干扰。

三档变异 (公平锁之争)
--------------------
  abs2.1   每基因组期望翻转数恒定 F=2.1 (v1/v2 用的锁)
  abs6.3   同锁, 压力 x3
  rel6pct  **相对压力恒定** p_flip=0.06 (= W=35 时的相对压力)。W=1050 时绝对破坏量
           变成 63 次翻转/基因组/代。若该档在大 W 崩掉而 abs2.1 不崩, 则说明
           **绝对翻转数才是正确的公平锁, 相对压力匹配是错的**。
  附带自检: rel6pct 与 abs2.1 在 W=35 处 p_flip 都是 0.06, 两条曲线必须**逐位重合**。

配对纪律: 所有条件/档在每个子代消耗的随机数数量与顺序完全相同 (PAIRED DRAWS),
逐种子可配对相减。param-only 的架构冻结 -> 其曲线与变异档无关, 只在基准档跑, 其余档复用。

用法:
  python scripts/exp_gens_mut_scale.py --quick      # 冒烟: 1 规模 x 2 档 x 1 种子 x 80 代
  python scripts/exp_gens_mut_scale.py              # 全量: 3 规模 x 3 档 x 3 条件 x 3 种子
  python scripts/exp_gens_mut_scale.py --resume     # 续跑 (rows.jsonl 增量)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_g_transfer import make_reference                                   # noqa: E402
from exp_scale_curve import Layout, N_TRUE_BIAS                             # noqa: E402
from exp_scale_curve_v2 import FastLayout, _arch_draw                        # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "exp_gens_mut_scale"
# 规模: W = 35 / 273 / 1050 (12 个真节点不变, 只加诱饵)
SCALES = [(4, 4, 3), (12, 12, 9), (24, 24, 18)]

P_REL = 0.06                 # = 2.1 / 35, 即 W=35 时的相对压力
ARMS: list[tuple[str, float | None]] = [
    ("abs2.1", 2.1),         # v1/v2 的公平锁
    ("abs6.3", 6.3),         # 3 倍变异压力
    ("rel6pct", None),       # 相对压力恒定 (公平锁证伪档)
]
BASE_ARM = "abs2.1"

# 条件 -> (架构可动, 参数可动, 偏置初值/是否钉住)
CONDS = ("naive-joint", "param-only", "naive-arch-pinned")
COND_CFG: dict[str, tuple[bool, bool, str]] = {
    "naive-joint":       (True,  True,  "zero"),
    "param-only":        (False, True,  "zero"),
    "naive-arch-pinned": (True,  False, "ref"),   # 偏置钉在参考偏置, 不参与突变
}
# 架构冻结 -> 与变异档无关, 只在基准档跑
ARM_FREE_CONDS = ("param-only",)

SEEDS = (1, 2, 3)
GENS = 800
POP = 60
P_ARCH_TRIGGER = 0.7
P_BIAS_HIT = 0.3
BIAS_STD = 1.2
FULL_SOLVE = 16
CHECKPOINTS = (10, 25, 50, 100, 200, 400, 800)


def ref_bias_vector() -> np.ndarray:
    """参考偏置 (锁的另一半): 节点 1,2,3,5,7,9 的非零偏置。"""
    b = np.zeros(N_TRUE_BIAS)
    b[[0, 1, 2, 4, 6, 8]] = [3, -2, 5, 1, -4, 2]
    return b


def evolve(
    layout: FastLayout,
    cond: str,
    seed: int,
    xs, ys,
    gens: int,
    flip_arm: float | None,
    ref_bias12: np.ndarray,
    pop_size: int = POP,
    checkpoints: tuple[int, ...] = CHECKPOINTS,
) -> dict:
    """单条件单种子单变异档。在检查点记录当前最优/均值。"""
    arch_movable, param_movable, bias_start = COND_CFG[cond]
    p_flip = (flip_arm / layout.W) if flip_arm is not None else P_REL

    rng = random.Random(seed * 131 + 7)
    nprng = np.random.default_rng(seed)

    start = layout.ref_mask.copy()                      # 三条件同起点: 参考边集 (满解结构)
    bias0 = ref_bias12.copy() if bias_start == "ref" else np.zeros(N_TRUE_BIAS)

    pop = [(start.copy(), bias0.copy()) for _ in range(pop_size)]
    fit = np.array([layout.solved(m, b, xs, ys) for m, b in pop], dtype=float)

    n_ref_total = int(layout.ref_mask.sum())
    best_at: dict[str, float] = {}
    mean_at: dict[str, float] = {}
    # 种群层面的参考边保留率 —— naive-arch-pinned 的**真正**破坏读数。
    # 只看 best 会被精英保留策略无条件护住 (起点即满解 -> best 恒 16), 看不到破坏;
    # 均值保留率才反映「整个种群被变异冲掉多少」。
    ret_at: dict[str, float] = {}

    def snap(gen: int) -> None:
        best_at[str(gen)] = float(fit.max())
        mean_at[str(gen)] = float(fit.mean())
        ret_at[str(gen)] = round(
            float(np.mean([int((m & layout.ref_mask).sum()) for m, _ in pop]) / n_ref_total), 4)

    snap(0)
    for gen in range(1, gens + 1):
        order = np.argsort(-fit)
        new_pop = [pop[order[0]], pop[order[1]]]      # 精英 2
        while len(new_pop) < pop_size:
            a = rng.randrange(pop_size)
            b = rng.randrange(pop_size)
            if fit[a] < fit[b]:
                a = b
            c = rng.randrange(pop_size)
            d = rng.randrange(pop_size)
            if fit[c] < fit[d]:
                c = d
            ma, ba = pop[a]
            mc, bc = pop[c]
            # ---- PAIRED DRAWS: 数量与顺序对全部条件/全部档一致 ----
            # 架构层用 _arch_draw (固定长度块 + 切片), 见 exp_scale_curve_v2 的说明:
            # 直接写 nprng.random(layout.W) 会让流位置随 W 漂移, 于是「同种子配对」
            # 只在规模内部成立, 跨规模的比较退化成两条无关轨迹的对比。
            cx_mask = _arch_draw(nprng, layout.W) < 0.5
            cx_bias = nprng.random(N_TRUE_BIAS) < 0.5
            arch_trigger = rng.random()
            flip_draw = _arch_draw(nprng, layout.W)
            bias_hit = nprng.random(N_TRUE_BIAS) < P_BIAS_HIT
            bias_noise = nprng.normal(0, BIAS_STD, N_TRUE_BIAS)
            # ------------------------------------------------------
            cm = np.where(cx_mask, ma, mc)
            cb = np.where(cx_bias, ba, bc)
            if arch_movable and arch_trigger < P_ARCH_TRIGGER:
                cm = cm ^ (flip_draw < p_flip)
            if param_movable:
                cb = cb + bias_noise * bias_hit
            new_pop.append((cm, cb))
        pop = new_pop
        fit = np.array([layout.solved(m, b, xs, ys) for m, b in pop], dtype=float)
        if gen in checkpoints:
            snap(gen)

    best_i = int(np.argmax(fit))
    bm, _bb = pop[best_i]
    ref_on = int((bm & layout.ref_mask).sum())
    return {
        "cond": cond,
        "seed": seed,
        "W": int(layout.W),
        "flip_arm": "rel6pct" if flip_arm is None else f"abs{flip_arm}",
        "p_flip": round(float(p_flip), 6),
        "flips_expected_per_genome": round(float(p_flip * layout.W), 3),
        "gens": int(gens),
        "best_at": best_at,
        "mean_at": mean_at,
        "retention_at": ret_at,
        "final_best": float(fit.max()),
        "final_mean": float(fit.mean()),
        "final_pop_retention": ret_at[str(max(checkpoints))],
        "arch_progress": round(ref_on / n_ref_total, 4),
        "n_on": int(bm.sum()),
        "n_decoys_on": int((bm & ~layout.ref_mask).sum()),
    }


# ---------- 判定 ----------

def crossover(diff: dict[str, float], checkpoints) -> dict:
    """diff = naive-joint − param-only。交叉点 = 第一个满足「该点 > 0 且此后全部 > 0」的点。

    要求「持续反超」, 避免把单点噪声当交叉。
    """
    keys = [str(c) for c in checkpoints]
    known = [(k, diff[k]) for k in keys if k in diff and diff[k] is not None]
    cross = None
    for i, (k, v) in enumerate(known):
        if v > 0 and all(w > 0 for _, w in known[i:]):
            cross = int(k)
            break
    return {
        "diff_by_gen": {k: round(v, 3) for k, v in known},
        "crossover_gen": cross,
        "sign_at_last": ("+" if known and known[-1][1] > 0 else "-") if known else None,
        "min_diff": round(min(v for _, v in known), 3) if known else None,
        "max_diff": round(max(v for _, v in known), 3) if known else None,
    }


def verdict_of(per: dict[str, dict], ws: list[int], last_gen: int) -> dict:
    """预注册判定 (先写规则再跑)。

    A. **时间竞争成立**: 基准档存在交叉点, 且交叉世代数随 W 单调不减。
       (None = 800 代内不交叉, 视作 +inf)
    B. **变异率放大破坏**: 同一 W 下 abs6.3 的交叉点不早于 abs2.1。
    C. **公平锁证伪**: rel6pct 在最大 W 处的末点比 abs2.1 低 >= 3 分。
    D. **自检**: W=35 处 rel6pct 与 abs2.1 的 p_flip 相同 且 曲线逐点重合。
    """
    def g(W: int, arm: str, field: str, gen: int | None = None):
        e = per[f"{W}|{arm}"]
        if gen is None:
            return e[field]
        return e[field][str(gen)]

    cross_by = {
        arm: {str(W): g(W, arm, "crossover")["crossover_gen"] for W in ws}
        for arm in ("abs2.1", "abs6.3", "rel6pct")
    }

    w0, wN = ws[0], ws[-1]
    same_p = abs(g(w0, "rel6pct", "mean_p_flip") - g(w0, "abs2.1", "mean_p_flip")) < 1e-12
    keys0 = sorted(g(w0, "rel6pct", "mean_best_at_naive-joint"), key=int)
    same_curve = all(
        abs(g(w0, "rel6pct", "mean_best_at_naive-joint", int(k))
            - g(w0, "abs2.1", "mean_best_at_naive-joint", int(k))) < 1e-9
        for k in keys0
    )

    seq = [cross_by["abs2.1"][str(W)] for W in ws]
    norm = [x if x is not None else 10 ** 9 for x in seq]
    n_cross = sum(1 for x in seq if x is not None)
    a_ok = bool(n_cross >= 2 and all(norm[i] >= norm[i - 1] for i in range(1, len(norm))))

    b_ok: bool | None = True
    b_n = 0                               # 真正能比较的 W 个数 (基准档有交叉点)
    for W in ws:
        x1, x2 = cross_by["abs2.1"][str(W)], cross_by["abs6.3"][str(W)]
        if x1 is None:
            continue                      # 基准档都不交叉, 该项无信息
        b_n += 1
        if x2 is None or x2 >= x1:
            continue                      # 高变异率不交叉 / 更晚 = 符合预测
        b_ok = False
    if b_n == 0:
        # 一个交叉点都没有时, 「高变异率不更早交叉」是**空真** —— 没有东西可以排序。
        # 报 true 会让人以为 B 被证实了 (首版就是这样: A=false 但 B=true 并列打印)。
        # 无信息就必须报无信息, 不能报「符合预测」。
        b_ok = None

    rel_last = g(wN, "rel6pct", "mean_best_at_naive-joint", last_gen)
    abs_last = g(wN, "abs2.1", "mean_best_at_naive-joint", last_gen)
    c_ok = bool(abs_last - rel_last >= 3.0)

    if not (same_p and same_curve):
        v = "SELFCHECK_FAILED"
    elif a_ok and b_ok is True and c_ok:
        v = "TIME_COMPETITION_CONFIRMED"
    elif a_ok:
        v = "TIME_COMPETITION_PARTIAL"
    else:
        v = "TIME_COMPETITION_NOT_OBSERVED"

    return {
        "verdict": v,
        "crossover_gen_by_arm_and_W": cross_by,
        "A_crossover_shifts_with_W": a_ok,
        "B_higher_mutation_delays_crossover": b_ok,
        "B_comparable_W_count": b_n,
        "C_relative_pressure_arm_collapses_at_large_W": c_ok,
        "selfcheck_rel6pct_equals_abs21_at_W35": {"same_p_flip": same_p, "same_curve": same_curve},
        "rel_vs_abs_best_at_largest_W": {
            "rel6pct": rel_last, "abs2.1": abs_last, "gap": round(abs_last - rel_last, 3)},
        "rule": (
            "SELFCHECK_FAILED: W 最小时 rel6pct 与 abs2.1 的 p_flip 或曲线不一致; "
            "TIME_COMPETITION_CONFIRMED: 基准档交叉点随 W 单调不减 且 高变异率交叉不更早 "
            "且 rel6pct 在最大 W 落后 abs2.1 >=3 分; PARTIAL: 仅 A 成立; 否则 NOT_OBSERVED。"
            "B 在无交叉点时返回 null (空真不等于证实)"
        ),
    }


# ---------- 并行 ----------

_G: dict = {}


def _init(xs, ys, rb, gens, pop_size, checkpoints) -> None:
    _G.update(xs=xs, ys=ys, rb=rb, gens=gens, pop=pop_size, ck=checkpoints, layouts={})


def _task(job: tuple) -> dict:
    """一个 (规模, 条件, 种子, 变异档) 的运行单元。布局在进程内按需缓存。"""
    w, cond, seed, arm_label, arm_F = job
    lay = _G["layouts"].get(w)
    if lay is None:
        lay = FastLayout(*w)
        _G["layouts"][w] = lay
    r = evolve(lay, cond, seed, _G["xs"], _G["ys"], _G["gens"], arm_F, _G["rb"],
               _G["pop"], _G["ck"])
    r["w1w2w3"] = list(w)
    r["arm_requested"] = arm_label
    r["arm_used"] = "rel6pct" if arm_F is None else f"abs{arm_F}"
    return r


def main() -> None:
    ap = argparse.ArgumentParser(
        description="世代数 x 变异率 x 规模 三因子交互扫描 (P1.6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="冒烟: 1 规模 x 2 档 x 1 种子 x 80 代")
    ap.add_argument("--gens", type=int, default=GENS)
    ap.add_argument("--pop", type=int, default=POP)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--arms", default=None, help="覆盖变异档, 如 abs2.1,rel6pct")
    ap.add_argument("--scales", default=None, help="覆盖规模, 如 4,4,3;24,24,18")
    ap.add_argument("--resume", action="store_true", help="读 rows.jsonl 续跑")
    ap.add_argument("--out", default=None, help="输出目录 (默认 results/exp_gens_mut_scale)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    global OUT
    if args.out:
        OUT = Path(args.out)

    if args.quick:
        scales, seeds, gens = SCALES[:1], (1,), 80
        arms = [("abs2.1", 2.1), ("rel6pct", None)]
        checkpoints = (10, 25, 50, 80)
    else:
        scales, seeds, gens = SCALES, tuple(int(s) for s in args.seeds.split(",")), args.gens
        arms = ARMS
        checkpoints = tuple(c for c in CHECKPOINTS if c <= gens)
        if gens not in checkpoints:
            checkpoints = checkpoints + (gens,)

    if args.scales:
        scales = [tuple(int(x) for x in seg.split(",")) for seg in args.scales.split(";")]
    if args.arms:
        want = {s.strip() for s in args.arms.split(",")}
        arms = [(lab, f) for lab, f in ARMS if lab in want]
    pop_size = args.pop

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _ref_mask, _ref_bias, xs, ys = make_reference()
    rb = ref_bias_vector()

    # ---- 等价性自检: 真 Layout vs FastLayout (两个不同对象, 不是自己跟自己比) ----
    w_probe = scales[0]
    slow, fast = Layout(*w_probe), FastLayout(*w_probe)
    rng = np.random.default_rng(20260918)
    mismatch = 0
    for _ in range(4):
        m = rng.random(fast.W) < 0.3
        b = rng.normal(0, 2.0, N_TRUE_BIAS)
        if not np.array_equal(slow.run(m, b, xs), fast.run(m, b, xs)):
            mismatch += 1
    if not np.array_equal(slow.run(fast.ref_mask.copy(), rb, xs),
                          fast.run(fast.ref_mask.copy(), rb, xs)):
        mismatch += 1
    eq = {"W": int(fast.W), "mismatch": int(mismatch), "ok": mismatch == 0}
    print(f"等价性自检 (Layout vs FastLayout): {eq}", flush=True)

    # ---- 起点读数: 必须「参考边集 + 参考偏置 = 满解」, 否则实验无意义 ----
    start_ref_bias, start_zero_bias, W_of = {}, {}, {}
    for w in scales:
        lay = FastLayout(*w)
        W_of[w] = int(lay.W)
        start_ref_bias[str(lay.W)] = int(lay.solved(lay.ref_mask, rb, xs, ys))
        start_zero_bias[str(lay.W)] = int(lay.solved(lay.ref_mask, np.zeros(N_TRUE_BIAS), xs, ys))
    print(f"起点: 参考边集+参考偏置 = {start_ref_bias} (应全为 {FULL_SOLVE})", flush=True)
    print(f"起点: 参考边集+零偏置   = {start_zero_bias}", flush=True)
    if any(v != FULL_SOLVE for v in start_ref_bias.values()):
        print("!! 警告: 参考偏置未达满解, naive-arch-pinned 的起点不是 16/16", flush=True)

    # ---- 任务清单 ----
    rows_path = OUT / "rows.jsonl"
    rows: list[dict] = []
    done: set[tuple] = set()
    stale: list[tuple] = []
    if args.resume and rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            # 协议不同的记录不能复用 —— 续跑键只写 (W,arm,cond,seed) 时, 一次
            # `--quick`(80 代)冒烟会把低功效结果永久留下, 全量跑静默继承。
            # 同一个坑在 exp_scale_curve_v2 里已经中过一次。
            if r.get("gens") != gens or r.get("pop") != pop_size:
                stale.append((r.get("gens"), r.get("pop")))
                continue
            rows.append(r)
            done.add((r["W"], r["flip_arm"], r["cond"], r["seed"]))
        if stale:
            print(f"!! 忽略 {len(stale)} 条协议不同的旧记录 "
                  f"(旧 gens/pop = {sorted(set(stale))}; 本次 gens={gens} pop={pop_size}); "
                  f"它们不会被复用", flush=True)
        print(f"续跑: 已有 {len(rows)} 条记录", flush=True)

    jobs: list[tuple] = []
    for w in scales:
        for arm_label, arm_F in arms:
            for cond in CONDS:
                eff = (BASE_ARM, dict(ARMS)[BASE_ARM]) if cond in ARM_FREE_CONDS else (arm_label, arm_F)
                for seed in seeds:
                    if (W_of[w], eff[0], cond, seed) in done:
                        continue
                    jobs.append((w, cond, seed, arm_label, eff[1]))

    print(f"待跑 {len(jobs)} 个运行单元 (workers={args.workers})", flush=True)
    if jobs:
        from multiprocessing import Pool
        with Pool(args.workers, initializer=_init,
                  initargs=(xs, ys, rb, gens, pop_size, checkpoints)) as pool:
            for r in pool.imap_unordered(_task, jobs):
                rows.append(r)
                done.add((r["W"], r["flip_arm"], r["cond"], r["seed"]))
                with rows_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                lg = str(max(checkpoints))
                print(f"W={r['W']:<5} arm={r['arm_used']:<8} {r['cond']:<18} s{r['seed']}: "
                      f"F/g={r['flips_expected_per_genome']:<7.1f} "
                      f"best@0={r['best_at']['0']:.0f} "
                      f"best@{lg}={r['best_at'][lg]:.0f} "
                      f"ret@{lg}={r['retention_at'][lg]:.3f}", flush=True)

    # ---- 汇总 ----
    arm_labels = [a for a, _ in arms]
    per: dict[str, dict] = {}
    for W in sorted({r["W"] for r in rows}):
        for arm in sorted({r["arm_used"] for r in rows if r["W"] == W}):
            sub = [r for r in rows if r["W"] == W and r["arm_used"] == arm]
            if not sub:
                continue
            e: dict = {"n_per_cond": {}, "arm": arm, "W": int(W)}
            for cond in CONDS:
                c = [r for r in sub if r["cond"] == cond]
                if not c:
                    continue
                e["n_per_cond"][cond] = len(c)
                e[f"mean_best_at_{cond}"] = {
                    k: round(float(np.mean([r["best_at"][k] for r in c])), 3) for k in c[0]["best_at"]}
                e[f"mean_mean_at_{cond}"] = {
                    k: round(float(np.mean([r["mean_at"][k] for r in c])), 3) for k in c[0]["mean_at"]}
                e[f"mean_retention_at_{cond}"] = {
                    k: round(float(np.mean([r["retention_at"][k] for r in c])), 4)
                    for k in c[0]["retention_at"]}
                e[f"arch_progress_{cond}"] = round(float(np.mean([r["arch_progress"] for r in c])), 4)
                e[f"final_best_{cond}"] = round(float(np.mean([r["final_best"] for r in c])), 3)
            e["mean_p_flip"] = round(float(np.mean([r["p_flip"] for r in sub])), 6)
            e["flips_per_genome"] = round(float(np.mean([r["flips_expected_per_genome"] for r in sub])), 3)
            per[f"{W}|{arm}"] = e

    # param-only 只在基准档跑 -> 其余档按构造借用 (架构冻结, 与变异档无关)
    for W in sorted({r["W"] for r in rows}):
        base = per.get(f"{W}|{BASE_ARM}")
        if not base:
            continue
        for arm in {r["arm_used"] for r in rows if r["W"] == W}:
            e = per[f"{W}|{arm}"]
            if "mean_best_at_param-only" not in e:
                for k, v in base.items():
                    if k.startswith(("mean_best_at_param-only", "mean_mean_at_param-only",
                                     "arch_progress_param-only", "final_best_param-only")):
                        e[k] = v
                e["param_only_borrowed_from"] = BASE_ARM

    # 交叉点
    for key, e in per.items():
        nb, pb = e.get("mean_best_at_naive-joint"), e.get("mean_best_at_param-only")
        if nb and pb:
            e["crossover"] = crossover({k: nb[k] - pb[k] for k in nb}, [int(k) for k in nb])

    ws = sorted({r["W"] for r in rows})
    last_gen = max(checkpoints)

    # ---- 自检: param-only 必须跨规模逐位相同 ----
    # param-only 的起点是**参考边集 + 零偏置**且架构冻结 -> 适应度只依赖 bias,
    # 与 W 无关; 加上 PAIRED DRAWS 改用固定块抽样后, 噪声序列也与 W 无关 =>
    # **整条轨迹必须与 W 无关**。它与 exp_scale_curve_v2 的 directed 是同构臂,
    # 两者是这一条不变量在两个脚本里的两个实例。
    # 首版数据里三条 W 的 param-only 曲线明显不同 (10 代处 12.333 / 12.0 / 13.0),
    # 直接暴露了 nprng.random(layout.W) 造成的跨规模 RNG 流漂移。
    po_by_seed: dict[int, dict[int, dict]] = {}
    for r in rows:
        if r["cond"] == "param-only" and r["flip_arm"] == BASE_ARM:
            po_by_seed.setdefault(r["seed"], {})[r["W"]] = r["best_at"]
    po_identical = bool(po_by_seed) and all(
        all(po_by_seed[s].get(W) == po_by_seed[s].get(ws[0]) for W in ws)
        for s in po_by_seed for W in ws)
    po_diffs = {
        str(s): {str(W): po_by_seed[s].get(W) for W in ws} for s in sorted(po_by_seed)
    }
    print(f"自检 param-only 跨规模逐位相同: {po_identical} "
          f"(逐种子 {[(s, all(po_by_seed[s].get(W) == po_by_seed[s].get(ws[0]) for W in ws)) for s in sorted(po_by_seed)]})",
          flush=True)
    need = {"abs2.1", "abs6.3", "rel6pct"}
    have = {arm for W in ws for arm in
            (a for a in arm_labels if f"{W}|{a}" in per and "crossover" in per[f"{W}|{a}"])}
    if need <= have:
        vd = verdict_of(per, ws, last_gen)
    else:
        vd = {
            "verdict": "INCOMPLETE_GRID",
            "missing_arms": sorted(need - have),
            "crossover_gen_by_arm_and_W": {
                a: {str(W): per[f"{W}|{a}"]["crossover"]["crossover_gen"] for W in ws}
                for a in sorted(have) if all(f"{W}|{a}" in per and "crossover" in per[f"{W}|{a}"] for W in ws)
            },
            "rule": "三档 (abs2.1/abs6.3/rel6pct) 不全, 判定降级为 INCOMPLETE_GRID",
        }

    summary = {
        "protocol": (
            "三因子交互: 规模 W x 变异档 x 世代数。三条件同起点 = 参考边集 + "
            "零偏置(naive-joint / param-only) 或 参考偏置(naive-arch-pinned, 起点即满解)。"
            "naive-joint 架构+参数自由; param-only 架构冻结; naive-arch-pinned 架构自由但"
            "参数钉在参考偏置(纯破坏读数)。变异档 abs2.1/abs6.3 = 每基因组期望翻转数恒定 "
            "(p_flip=F/W); rel6pct = 相对压力恒定 (p_flip=0.06)。"
            f"检查点 {list(checkpoints)} 代, 跑到 {gens} 代; pop {pop_size}; "
            "全部条件/档共享 RNG 轨迹(逐种子配对)。"
        ),
        "equivalence_check": eq,
        "start_solved_with_ref_bias": start_ref_bias,
        "start_solved_with_zero_bias": start_zero_bias,
        "checkpoints": list(checkpoints),
        "gens": gens,
        "seeds": list(seeds),
        "arms": sorted(have),
        "per_W_arm": per,
        # param-only 跨规模逐位相同 =「跨规模 RNG 配对」+「诱饵中性」的双重不变量
        "param_only_scale_invariant": po_identical,
        "param_only_per_seed_by_W": po_diffs,
        "wall_min": round((time.time() - t0) / 60, 1),
        **vd,
    }
    if not po_identical:
        # 方法学失效要抢在结论之前返回: 跨规模差异此时无法归因于 W
        summary["verdict"] = "RNG_PAIRING_FAILED"
    (OUT / "gens_mut_scale_verdict.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 曲线图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        style = {
            "naive-joint": ("naive-joint (arch+param)", "o-"),
            "param-only": ("param-only (arch frozen)", "^-"),
            "naive-arch-pinned": ("naive-arch-pinned (pure destruction)", "s--"),
        }
        arms_plot = sorted(have)
        fig, axes = plt.subplots(2, len(arms_plot),
                                 figsize=(4.6 * len(arms_plot), 7.6), squeeze=False)
        for col, arm in enumerate(arms_plot):
            for row, (field, ylab, ylim) in enumerate((
                    ("mean_best_at", "best solved / 16", (-0.5, 16.5)),
                    ("mean_retention_at", "population ref-edge retention", (-0.05, 1.05)))):
                ax = axes[row][col]
                for cond, (label, sty) in style.items():
                    for W in ws:
                        e = per.get(f"{W}|{arm}")
                        key = f"{field}_{cond}"
                        if not e or key not in e:
                            continue
                        d = e[key]
                        gx = sorted(int(k) for k in d)
                        ax.plot(gx, [d[str(x)] for x in gx], sty,
                                label=f"{label} W={W}", alpha=0.85)
                ax.set_xscale("log")
                ax.set_xlabel("generation (log)")
                ax.set_ylabel(ylab)
                ax.set_ylim(*ylim)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=6)
                if row == 0:
                    ax.set_title(f"arm = {arm}")
        fig.tight_layout()
        fig.savefig(OUT / "gens_mut_scale_curve.png", dpi=130)
        summary["plot"] = "gens_mut_scale_curve.png"
        (OUT / "gens_mut_scale_verdict.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print("plot skip:", e)

    print()
    print(json.dumps(vd, ensure_ascii=False, indent=1))
    print("WROTE", OUT / "gens_mut_scale_verdict.json")


if __name__ == "__main__":
    main()
