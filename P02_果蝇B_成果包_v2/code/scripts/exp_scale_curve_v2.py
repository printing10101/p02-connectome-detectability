#!/usr/bin/env python
"""尺度插值 v2: 「缺失边找回」—— 把「发现」与「修复」解耦的达成率实验 (任务 P1.5).

v1 (exp_scale_curve.py) 的读数错在哪
------------------------------------
v1 用 naive - param-only 的**分数差**当「架构红利」。两处混杂:

1. **基线抽签**。param-only 从**随机 9 条边**出发且架构冻结, 它的分数是抽签结果
   (W=35 时 per-seed = [5, 12, 13], 跨度 8 分)。分数差被基线方差主导, W=35 处
   甚至算出负红利 -2.33 —— 读数在说「这一抽运气好」, 不是在说「架构搜索有没有用」。
2. **破坏与发现混在一起**。p_flip = 2.1/W 保证「每基因组期望翻转数恒定」, 于是小 W 下
   单条边的被翻概率大 (W=35 → 0.06)。参考边集里 14 条真边每代被翻掉的期望是 0.84 条,
   而翻开诱饵边毫无收益 —— 小 W 的 naive 主要在**破坏**已有的好结构。W 从 35 加到 2331,
   单边被翻概率降到 0.0009, 破坏变少, 分数自然上升。

两条叠在一起, v1 得到 DECAY_NOT_OBSERVED 是必然的。但这不是三基质定律错了, 是读数错了:
v1 测的是「给定随机起点, 自由架构突变净收益多少」, 而定律问的是
**「选择能不能在 W 个候选里看见该补的那一条边」**。

v2 的修正
---------
1. **同起点**: 所有条件都从**同一个缺陷结构**出发 —— 参考边集去掉 1 条关键边
   (关键边由单边消融校准选出)。基线不再抽签。
2. **层隔离**: 五个条件把「架构层」与「参数层」分开测, 不再用两层相减:
      naive-arch   架构自由 + **参数钉在参考偏置**  → 纯「发现」读数 (适应度 = 掩码的函数)
      param-only   架构冻结在缺陷结构 + 参数自由    → 纯「修复」读数
      directed     架构=参考边集 + 参数自由          → 天花板 (结构给定, 只需调参)
      naive-joint  架构自由 + 参数自由               → 现实情形 (与 v1 可比)
      sham         架构自由 + 参数钉住 + 适应度打乱  → 漂移地板
   参数钉住这一步是 v2 的关键: 它让 `final_best >= 16` 与「参考边集被完整找回」
   互为充要, 漂移无法伪造 (v2 第一版没做, sham 在 W=35 上拿到了 0.5 的假找回率)。
3. **配对**: 五条件共用同一 RNG 流 (每个子代消耗**相同数量**的随机数, 见 evolve 注释)。
4. **达成率为主读数**: 达标率是二值量, 不被基线方差淹没。找回率作独立佐证。

预测: naive-arch 的达标率与找回率随 W 单调衰减 —— 候选越多, 该补的那条边越难被选择看见;
param-only 的找回率恒为 0 (架构冻结), 且其终局分数与 W 无关 (它根本没用候选池)。

用法:
  python scripts/exp_scale_curve_v2.py --quick          # 冒烟: 2 规模 x 1 种子 x 60 代
  python scripts/exp_scale_curve_v2.py                  # 全量: 4 规模 x 5 条件 x 2 缺边 x 3 种子
  python scripts/exp_scale_curve_v2.py --seeds 1,2,3,4,5,6,7,8,9,10 --drops 1 \
      --conds naive-arch,directed,sham --out results/tj_a4_scale/scale_v2
                                                        # TJ-A4 三臂协议 (TJ_A4_scale_axis)
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
from exp_g_transfer import OPS, DOMAIN, make_reference                      # noqa: E402
from exp_scale_curve import Layout, SCALES, FLIPS_PER_GENOME, N_TRUE_BIAS   # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "exp_scale_curve_v2"

# 条件表: (架构是否可动, 参数是否可动, 适应度是否打乱, 起始结构)
#   起始结构 "defect" = 参考边集去 1 条关键边; "ref" = 参考边集
CONDS: dict[str, dict] = {
    "naive-arch":  {"arch": True,  "param": False, "shuffle": False, "start": "defect"},
    "param-only":  {"arch": False, "param": True,  "shuffle": False, "start": "defect"},
    "directed":    {"arch": False, "param": True,  "shuffle": False, "start": "ref"},
    "naive-joint": {"arch": True,  "param": True,  "shuffle": False, "start": "defect"},
    "sham":        {"arch": True,  "param": False, "shuffle": True,  "start": "defect"},
}
COND_ORDER = ("naive-arch", "param-only", "directed", "naive-joint", "sham")

SEEDS = (1, 2, 3)
GENS = 300
POP = 60
P_ARCH_TRIGGER = 0.7      # 架构突变触发概率 (与 v1 一致)
P_BIAS_HIT = 0.3          # 参数突变命中率 (与 v1 一致)
BIAS_STD = 1.2
N_DROPS = 2               # 取最关键的前 2 条参考边各做一个子条件
FULL_SOLVE = 16


# ---------- 跨规模 RNG 配对: 抽样长度必须与 W 无关 ----------

# 每子代的架构层抽样**必须从固定长度的块里取**, 再截到 [:W]。
#
# 原版直接 `nprng.random(layout.W)`: 由于 nprng 是逐 seed 复位的单条流, 每次抽样
# 推进流的位置是 W, 于是**同一 seed 在不同规模上抽到的是完全不同的噪声** ——
# W=35 与 W=273 的第 3 个子代 bias_noise 毫无关系 (实测 [:4] = [1.4715, 1.1546,
# -3.2535, 0.05] vs [1.0845, 0.1647, 0.5956, -0.4209])。
#
# 后果: 「同种子配对」只在**规模内部**成立 (五条件同 W, 流位置一致), 跨规模比较
# 则是两条无关轨迹的对比。P1.5 的 directed 臂给出了这个混杂的干净读数 —— 它
# 的结构给定、适应度是 bias 的纯函数且**跨四规模逐位相同** (实测 5 组随机 bias
# 全部一致, 参考偏置 16/16、零偏置 1/16 亦全规模一致), 所以它的跨规模差异
# 100% 来自噪声; 实测却是 0.3333 / 1.0000 / 0.6667 / 0.6667。
#
# 修法: 固定块 + 切片。块长取全规模最大 W, 并加断言 —— 若日后加了更大的规模
# 却忘了抬高块长, 配对会**静默**失效, 所以宁可在这里响亮地报错。
ARCH_DRAW_BLOCK = max(Layout(*s).W for s in SCALES)


def _arch_draw(nprng: np.random.Generator, w: int) -> np.ndarray:
    """从固定长度块里抽 w 个数。流位置只与块长有关, 与 w 无关。"""
    if w > ARCH_DRAW_BLOCK:
        raise ValueError(
            f"W={w} 超过 ARCH_DRAW_BLOCK={ARCH_DRAW_BLOCK}: 跨规模 RNG 配对会失效。"
            f"请把 ARCH_DRAW_BLOCK 抬到 >= {w} (通常改成 SCALES 里最大规模的 W)。")
    return nprng.random(ARCH_DRAW_BLOCK)[:w]


# ---------- 快速基质 (与 Layout 语义等价, 预建入边表) ----------

class FastLayout(Layout):
    """同 Layout, 但把「每节点遍历全部层内边」换成预建入边索引。

    Layout.run 对每个节点扫一遍该层所有边 (W=2331 时约 9.5 万次内层迭代),
    FastLayout 只扫该节点的入边 (约 W 次)。语义必须与 Layout 完全一致,
    由 self_check 逐位对读保证。
    """

    def __init__(self, *w: int):
        super().__init__(*w)
        in_edges: dict[int, list[tuple[int, int]]] = {n: [] for n in range(self.n_nodes)}
        for ei, (a, b) in enumerate(self.edges):
            in_edges[b].append((ei, a))
        self._in = {
            n: (np.array([ei for ei, _ in v], dtype=int),
                np.array([a for _, a in v], dtype=int))
            for n, v in in_edges.items()
        }
        # 求值顺序必须沿层推进 (诱饵节点与真节点同层)
        self.eval_order = list(self.s1) + list(self.s2) + list(self.s3) + [self.out]

    def run(self, mask: np.ndarray, bias12: np.ndarray, xs: np.ndarray) -> np.ndarray:
        full = self.full_biases(bias12)
        vals = np.zeros((self.n_nodes, len(xs)), dtype=np.int64)
        vals[0] = xs.astype(np.int64)
        for node in self.eval_order:
            eis, srcs = self._in[node]
            sel = mask[eis] if len(eis) else np.zeros(0, dtype=bool)
            # 空选择时 vals[[]] 形状 (0, n_tasks), sum(axis=0) 仍给出 n_tasks 个 0
            acc = vals[srcs[sel]].sum(axis=0)
            vals[node] = OPS[(node - 1) % 4](np.round(acc + full[node]) % DOMAIN)
        return vals[self.out].astype(float)


def ref_bias_vector() -> np.ndarray:
    """参考偏置 (11 维真内节点偏置)。

    单独抽成函数而不是在 main 里就地构造: P1.6 (exp_gens_mut_scale) 要用同一个
    参考偏置, 两边各写一份迟早会漂开 —— 而「参考解」一旦漂开, 所有以它为
    起点的条件都失去可比性, 且不会有任何报错。
    """
    b = np.zeros(N_TRUE_BIAS)
    b[[0, 1, 2, 4, 6, 8]] = [3, -2, 5, 1, -4, 2]   # 节点 1,2,3,5,7,9 的偏置
    return b


def self_check(layout: Layout, fast: FastLayout, ref_bias12, xs) -> dict:
    """FastLayout 与 Layout 的等价性对读 (随机掩码 + 参考边集)。"""
    rng = np.random.default_rng(20260918)
    bad = 0
    for _ in range(6):
        m = rng.random(layout.W) < 0.3
        b = rng.normal(0, 2.0, N_TRUE_BIAS)
        if not np.array_equal(layout.run(m, b, xs), fast.run(m, b, xs)):
            bad += 1
    if not np.array_equal(layout.run(layout.ref_mask.copy(), ref_bias12, xs),
                          fast.run(layout.ref_mask.copy(), ref_bias12, xs)):
        bad += 1
    return {"W": int(layout.W), "mismatch": int(bad), "ok": bad == 0}


# ---------- 校准: 哪条参考边是关键的 ----------

def calibrate(layout: Layout, ref_bias12: np.ndarray, xs, ys) -> tuple[int, list[dict]]:
    """单边消融: 去掉每条参考边, 看解题数掉多少。掉得最多的 = 最关键的。

    这是 P0.1 纪律的尺度版: 先证明「值钱的架构改动存在」(缺边真的掉分),
    再问选择能不能找到它。缺边不掉分的话整个实验没有意义。

    **返回的是边的 (src,dst) 身份, 不是下标。** 下标在不同规模下指向不同的边
    (e_in 长度随 s1 增长), 用下标跨规模复用会把诱饵边当成缺边 —— 那组条件里
    参考边集根本没被动过, 所有条件都拿 16/16, 读数全是噪声。
    """
    full = layout.solved(layout.ref_mask, ref_bias12, xs, ys)
    rows = []
    for ei in np.flatnonzero(layout.ref_mask):
        m = layout.ref_mask.copy()
        m[ei] = False
        s = layout.solved(m, ref_bias12, xs, ys)
        rows.append({
            "ei": int(ei),
            "edge": [int(layout.edges[ei][0]), int(layout.edges[ei][1])],
            "solved_without": int(s),
            "drop": int(full - s),
        })
    rows.sort(key=lambda r: (-r["drop"], r["ei"]))
    return int(full), rows


def drop_indices(layout: Layout, drop_pairs: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """把缺边的 (src,dst) 身份翻译成**本规模下**的下标。"""
    out = []
    for p in drop_pairs:
        key = (int(p[0]), int(p[1]))
        if key not in layout.eindex:
            raise KeyError(f"缺边 {key} 不在 W={layout.W} 的候选边集里")
        out.append(int(layout.eindex[key]))
    return tuple(out)


# ---------- 主进化 ----------

def evolve(
    layout: FastLayout,
    cond: str,
    seed: int,
    xs, ys,
    drop_eis: tuple[int, ...],
    ref_bias12: np.ndarray,
    gens: int,
    pop_size: int,
) -> dict:
    """单条件单种子。

    **配对前提**: 所有条件在每个子代生成步骤里消耗的随机数**数量与顺序完全相同**
    (见下方标注 "PAIRED DRAWS")。条件只决定「抽出来的数怎么用」, 不决定抽几个。
    这样五条件共享同一条 RNG 轨迹, 逐种子可配对比较。
    """
    cfg = CONDS[cond]
    rng = random.Random(seed * 131 + 7)
    nprng = np.random.default_rng(seed)
    shuf_rng = np.random.default_rng(seed + 20_000)   # 只用于 sham 打乱, 不污染主轨迹

    start = layout.ref_mask.copy()
    if cfg["start"] == "defect":
        for ei in drop_eis:
            start[ei] = False
    bias0 = ref_bias12.copy() if not cfg["param"] else np.zeros(N_TRUE_BIAS)

    p_flip = FLIPS_PER_GENOME / layout.W

    pop = [(start.copy(), bias0.copy()) for _ in range(pop_size)]
    fit = np.array([layout.solved(m, b, xs, ys) for m, b in pop], dtype=float)

    first_recover_gen: int | None = None
    for gen in range(gens):
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
            # ---- PAIRED DRAWS: 数量与顺序对所有条件一致 ----
            # 架构层用 _arch_draw (固定块 + 切片): 流位置与 W 无关,
            # 跨规模也能逐种子配对。直接写 nprng.random(layout.W) 会让
            # 流位置随 W 漂移, 跨规模比较退化成两条无关轨迹。
            cx_mask = _arch_draw(nprng, layout.W) < 0.5
            cx_bias = nprng.random(N_TRUE_BIAS) < 0.5
            arch_trigger = rng.random()
            flip_draw = _arch_draw(nprng, layout.W)
            bias_hit = nprng.random(N_TRUE_BIAS) < P_BIAS_HIT
            bias_noise = nprng.normal(0, BIAS_STD, N_TRUE_BIAS)
            # -------------------------------------------------
            cm = np.where(cx_mask, ma, mc)
            cb = np.where(cx_bias, ba, bc)
            if cfg["arch"] and arch_trigger < P_ARCH_TRIGGER:
                cm = cm ^ (flip_draw < p_flip)
            if cfg["param"]:
                cb = cb + bias_noise * bias_hit
            new_pop.append((cm, cb))
        pop = new_pop
        fit = np.array([layout.solved(m, b, xs, ys) for m, b in pop], dtype=float)
        if cfg["shuffle"]:
            fit = shuf_rng.permutation(fit)

        bm = pop[int(np.argmax(fit))][0]
        if (first_recover_gen is None and all(bm[ei] for ei in drop_eis)
                and float(fit.max()) >= FULL_SOLVE):
            first_recover_gen = gen

    best_i = int(np.argmax(fit))
    bm, _bb = pop[best_i]
    ref_on = int((bm & layout.ref_mask).sum())
    n_ref_total = int(layout.ref_mask.sum())
    final_best = float(fit.max())
    edge_on = bool(all(bm[ei] for ei in drop_eis))
    return {
        "cond": cond,
        "seed": seed,
        "W": int(layout.W),
        # 协议必须写进记录。续跑键若不含协议, 一次 `--quick`(60 代)冒烟就会把
        # 低功效结果**永久**留在 rows.jsonl 里, 之后的 300 代全量跑会静默复用它们
        # (本轮就中了: `--quick` 先写进正式目录, 全量跑读了出来, 于是 W=35/273 的
        # directed s1 是 60 代的 15 分, W=1050/2331 才是 300 代的 16 分)。
        "gens": int(gens),
        "pop": int(pop_size),
        "drop_eis": [int(e) for e in drop_eis],
        "final_best": final_best,
        "final_mean": float(fit.mean()),
        # 原始读数: 缺边是否被打开。**单独用不可靠** —— 无选择时它趋近 0.5
        # (精英变成随机游走, 缺边开/关各半), 所以 sham 也能拿到 ~0.5。
        "edge_on": edge_on,
        # 有效「发现」: 缺边打开 **且** 解题数到顶。漂移拿不到 16/16, 因此
        # 这个读数能把「找回」与「偶然翻开」分开 —— 主判定用它。
        "recovered": bool(edge_on and final_best >= FULL_SOLVE),
        # 完整复原: 参考边集被完整找回 (比 final_best>=16 更严 —— 参考边集不唯一)
        "fully_restored": bool(ref_on == n_ref_total),
        "first_recover_gen": first_recover_gen,
        "n_on": int(bm.sum()),
        "n_ref_edges_kept": ref_on,
        "n_ref_edges_total": n_ref_total,
        # 架构进展 (连续读数, 比二值 recovered 更灵敏): 参考边集保住了多少
        "arch_progress": round(ref_on / n_ref_total, 4),
        "n_decoys_on": int((bm & ~layout.ref_mask).sum()),
    }


# ---------- 判定 ----------

def verdict_of(per_W: dict[int, dict], ws: list[int]) -> dict:
    """预注册判定规则 (先写规则再跑, 避免事后挑读数)。

    条件子集 (--conds, A4 三臂协议) 不构成五条件判定 —— 缺 param-only/naive-joint
    时下面的自检与注意力竞争读数无从算起, 与其输出 WIRING_ERROR 噪声, 不如响亮声明
    本脚本内建判定不适用, 主判定由调用方 (tj_run_a4_scale) 按冻结键自行计算。

    三条读数, 对应三个不同的问题:
      A. 纯架构搜索 (naive-arch) 能不能把缺边找回来? → achieved_rate / first_recover_gen
         随 W 的衰减 = 盲搜的覆盖难度。
      B. 参数层能修复缺陷结构吗? (param-only vs directed) → 与 W 无关是自检, 分数差 = 修复上限。
      C. **注意力竞争**: 把参数层加回来 (naive-joint) 之后, 架构层进展掉了多少?
         `attention_gap(W) = arch_progress(naive-arch) - arch_progress(naive-joint)`
         这是「选择可见性不对称」在尺度上的直接读数: 两层同时在场上时, 架构层被挤掉多少。
    """
    missing = [c for c in COND_ORDER if per_W[ws[0]].get(c, {}).get("n", 0) == 0]
    if missing:
        return {
            "verdict": "COND_SUBSET_NO_VERDICT",
            "missing_conds": missing,
            "rule": ("--conds 条件子集跑不构成 v2 五条件判定 "
                     "(缺自检臂 param-only / naive-joint); 主判定归调用方"),
        }
    rec = [per_W[W]["naive-arch"]["recovery_rate"] for W in ws]
    ach = [per_W[W]["naive-arch"]["achieved_rate"] for W in ws]
    shm = [per_W[W]["sham"]["achieved_rate"] for W in ws]
    par_rec = [per_W[W]["param-only"]["recovery_rate"] for W in ws]

    arch_prog = [per_W[W]["naive-arch"]["mean_arch_progress"] for W in ws]
    joint_prog = [per_W[W]["naive-joint"]["mean_arch_progress"] for W in ws]
    gap = [a - j for a, j in zip(arch_prog, joint_prog)]

    # 自检 0 (新, 最关键): **directed 的逐种子终局必须跨规模逐位相同**。
    #
    # directed 的架构被直装为参考边集, 诱饵边全关 -> 真节点收到的输入只来自参考边,
    # 求值顺序与 OPS 分配对真节点与 W 无关 => 适应度是 bias 的纯函数, 与 W 无关
    # (实测 5 组随机 bias 跨四规模逐位一致; 参考偏置 16/16、零偏置 1/16 亦全规模一致)。
    # 又因 PAIRED DRAWS 现在用固定块抽样, 噪声序列也与 W 无关。两者相加 =>
    # **整条进化轨迹必须与 W 无关**。
    #
    # 这条不变量很强, 所以它一箭双雕: 既验了「跨规模 RNG 配对修好了」, 也验了
    # 「诱饵节点确实是中性的」。首版数据里 directed 给出 0.3333/1.0/0.6667/0.6667,
    # 直接暴露了 RNG 流位置随 W 漂移的 bug。
    dir_runs = {str(W): per_W[W]["directed"]["per_run"] for W in ws}
    dir_keys = set(dir_runs[str(ws[0])])
    dir_identical = all(set(dir_runs[str(W)]) == dir_keys for W in ws) and all(
        dir_runs[str(ws[0])][k] == dir_runs[str(W)][k]
        for W in ws[1:] for k in dir_keys)
    dir_diffs = {str(W): {k: dir_runs[str(W)].get(k) for k in dir_keys} for W in ws}

    checks: dict[str, object] = {
        # 自检 0: directed 跨规模逐位相同 (RNG 配对 + 诱饵中性 的双重不变量)
        "directed_scale_invariant": bool(dir_identical),
        # 自检 1: param-only 架构冻结 -> 缺边永远打不开
        "param_only_edge_off": all(
            per_W[W]["param-only"]["edge_on_rate"] == 0.0 for W in ws),
        # 自检 2: directed 结构直装 -> 缺边恒为开
        "directed_edge_on": all(
            per_W[W]["directed"]["edge_on_rate"] == 1.0 for W in ws),
        # 漂移地板: sham 不得与 naive-arch 同量级
        "sham_below_naive": bool(shm[0] <= max(0.1, 0.34 * ach[0])),
        # 自检 3: param-only 不用候选池 -> 其分数跨 W 恒定
        "param_only_W_invariant": bool(
            max(per_W[W]["param-only"]["mean_final_best"] for W in ws)
            - min(per_W[W]["param-only"]["mean_final_best"] for W in ws) <= 1.0),
    }

    def violations(xs_: list[float], tol: float = 1e-9) -> int:
        return sum(1 for i in range(1, len(xs_)) if xs_[i] > xs_[i - 1] + tol)

    rec_bad, ach_bad = violations(rec), violations(ach)
    ratio = (ach[-1] / ach[0]) if ach[0] > 0 else None

    # C. 注意力竞争
    competition = bool(gap[-1] >= 0.25 and gap[-1] >= gap[0])
    # A. 纯架构盲搜的覆盖衰减
    blind_decay = bool(ach_bad == 0 and ratio is not None and ratio < 1 / 3)

    if not checks["directed_scale_invariant"]:
        # 这是**方法学**失效, 不是结论失效: 跨规模的差异无法归因于 W, 只能归因于
        # 两条无关的 RNG 轨迹。此时任何跨规模读数都不该被引用, 所以必须抢先返回。
        v = "RNG_PAIRING_FAILED"
    elif not (checks["param_only_edge_off"] and checks["directed_edge_on"]):
        v = "WIRING_ERROR"
    elif not checks["param_only_W_invariant"]:
        v = "WIRING_ERROR_PARAM_DRIFT"
    elif not checks["sham_below_naive"]:
        v = "SHAM_CONTAMINATED"
    elif competition:
        v = "ATTENTION_COMPETITION_CONFIRMED"
    elif blind_decay:
        v = "DISCOVERY_DECAY_CONFIRMED"
    elif ach_bad <= 1 and rec_bad <= 1:
        v = "DISCOVERY_DECAY_PARTIAL"
    else:
        v = "DISCOVERY_NOT_OBSERVED"

    return {
        "verdict": v,
        "checks": checks,
        "directed_per_run_by_W": dir_diffs,
        "achieved_nonincreasing_violations": f"{ach_bad}/{len(ach) - 1}",
        "recovery_nonincreasing_violations": f"{rec_bad}/{len(rec) - 1}",
        "achieved_last_over_first": round(ratio, 3) if ratio is not None else None,
        "arch_progress_arch_only_by_W": dict(zip(map(str, ws), arch_prog)),
        "arch_progress_joint_by_W": dict(zip(map(str, ws), joint_prog)),
        "attention_gap_by_W": dict(zip(map(str, ws), [round(g, 4) for g in gap])),
        "attention_competition": competition,
        "blind_search_decay": blind_decay,
        "rule": (
            "RNG_PAIRING_FAILED: directed 逐种子终局跨规模不逐位相同 "
            "(适应度在参考结构下与 W 无关, 故这是 RNG 配对失效或诱饵不中性); "
            "WIRING_ERROR: param-only 找回率非 0 或 directed 找回率非 1; "
            "WIRING_ERROR_PARAM_DRIFT: param-only 分数跨 W 变动 >1; "
            "SHAM_CONTAMINATED: sham 达标率 > max(0.1, naive-arch首点/3); "
            "ATTENTION_COMPETITION_CONFIRMED: 末点 attention_gap>=0.25 且 末点>=首点; "
            "DISCOVERY_DECAY_CONFIRMED: naive-arch 达标率逐点不增 且 末点<首点/3; "
            "PARTIAL: 达标率与找回率各最多一处反弹; 其余 NOT_OBSERVED"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="尺度插值 v2: 缺失边找回")
    ap.add_argument("--quick", action="store_true", help="冒烟: 2 规模 x 1 种子 x 60 代")
    ap.add_argument("--gens", type=int, default=GENS)
    ap.add_argument("--pop", type=int, default=POP)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--scales", default=None, help="覆盖规模, 如 4,4,3;12,12,9")
    ap.add_argument("--drops", type=int, default=N_DROPS)
    ap.add_argument("--conds", default=",".join(COND_ORDER),
                    help="条件子集 (逗号分隔); 子集时内建判定不适用 (见 verdict_of)")
    ap.add_argument("--out", default=None,
                    help="覆盖输出目录 (缺省 results/exp_scale_curve_v2); "
                         "TJ-A4 用独立目录避免协议混入历史 rows.jsonl")
    args = ap.parse_args()

    conds = tuple(c.strip() for c in args.conds.split(",") if c.strip())
    bad = [c for c in conds if c not in CONDS]
    if bad:
        raise SystemExit(f"未知条件: {bad} —— 可选: {COND_ORDER}")
    if not conds:
        raise SystemExit("--conds 为空")

    if args.quick:
        scales, seeds, gens, n_drops = SCALES[:2], (1,), 60, 1
    else:
        scales = SCALES
        seeds = tuple(int(s) for s in args.seeds.split(","))
        gens, n_drops = args.gens, args.drops
    pop_size = args.pop
    if args.scales:
        scales = [tuple(int(x) for x in seg.split(",")) for seg in args.scales.split(";")]

    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _ref_mask, _ref_bias, xs, ys = make_reference()
    ref_bias12 = ref_bias_vector()

    # ---- 阶段 0: 等价性自检 + 关键边校准 ----
    # 注意: 这里必须传**两个不同的对象** (真 Layout 与 FastLayout)。首版写成
    # self_check(base, base, ...) —— 两边都是 FastLayout, 等价于自己跟自己比,
    # 恒真。这种「自检永远通过」的写法比没有自检更危险: 它给出虚假的安全感。
    base = FastLayout(*SCALES[0])
    slow = Layout(*SCALES[0])
    eq = self_check(slow, base, ref_bias12, xs)
    full_solve, crit = calibrate(base, ref_bias12, xs, ys)
    # 缺边用 (src,dst) 身份, 不用下标 —— 下标跨规模指向不同的边
    drops = [tuple(c["edge"]) for c in crit[:n_drops]]
    print(f"等价性自检: {eq}", flush=True)
    print(f"参考满解 = {full_solve}/16; 单边消融前 5: "
          f"{[(tuple(c['edge']), c['drop']) for c in crit[:5]]}", flush=True)
    print(f"选用缺边(边身份): {drops}", flush=True)
    (out_dir / "edge_criticality.json").write_text(
        json.dumps({"reference_full_solve": full_solve, "single_edge_ablation": crit,
                    "dropped_for_main": [list(d) for d in drops]},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 主实验 (增量落盘, 中断可续) ----
    rows_path = out_dir / "rows.jsonl"
    done: set[tuple] = set()
    rows: list[dict] = []
    stale: list[tuple] = []
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            # 协议不同的记录**不能复用**。续跑键只写 (W,drop,cond,seed) 时,
            # `--quick`(60 代)冒烟会把低功效结果永久留在目录里, 全量跑静默继承,
            # 表现为「同一个不变量在部分规模上成立、部分不成立」——极难定位。
            if r.get("gens") != gens or r.get("pop") != pop_size:
                stale.append((r.get("gens"), r.get("pop")))
                continue
            rows.append(r)
            done.add((r["W"], tuple(r["drop"]), r["cond"], r["seed"]))
        if stale:
            print(f"!! 忽略 {len(stale)} 条协议不同的旧记录 "
                  f"(旧 gens/pop = {sorted(set(stale))}; 本次 gens={gens} pop={pop_size}); "
                  f"它们不会被复用", flush=True)
        print(f"续跑: 已有 {len(rows)} 条记录", flush=True)

    for w in scales:
        layout = FastLayout(*w)
        for d in drops:
            d_eis = drop_indices(layout, [d])   # 身份 -> 本规模下标
            for cond in conds:
                for seed in seeds:
                    key = (int(layout.W), tuple(int(x) for x in d), cond, seed)
                    if key in done:
                        continue
                    r = evolve(layout, cond, seed, xs, ys, d_eis, ref_bias12,
                               gens, pop_size)
                    r["w1w2w3"] = list(w)
                    r["drop"] = [int(x) for x in d]
                    r["drop_ei"] = [int(x) for x in d_eis]
                    rows.append(r)
                    done.add(key)
                    with rows_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print(f"W={layout.W:<5} drop={d} {cond:<12} "
                          f"s{seed}: best={r['final_best']:.0f}/16 "
                          f"rec={int(r['recovered'])} full={int(r['fully_restored'])} "
                          f"gen={r['first_recover_gen']}", flush=True)

    # ---- 汇总 (按 W 聚合, 跨缺边/种子平均; 条件子集时只聚合在场的条件) ----
    per_W: dict[int, dict] = {}
    for W in sorted({r["W"] for r in rows}):
        sub = [r for r in rows if r["W"] == W]
        agg: dict[str, dict] = {}
        for cond in conds:
            c = [r for r in sub if r["cond"] == cond]
            fgens = [r["first_recover_gen"] for r in c if r["first_recover_gen"] is not None]
            agg[cond] = {
                "n": len(c),
                "edge_on_rate": round(float(np.mean([r["edge_on"] for r in c])), 4),
                "recovery_rate": round(float(np.mean([r["recovered"] for r in c])), 4),
                "achieved_rate": round(float(np.mean([r["final_best"] >= FULL_SOLVE for r in c])), 4),
                "restored_rate": round(float(np.mean([r["fully_restored"] for r in c])), 4),
                "mean_final_best": round(float(np.mean([r["final_best"] for r in c])), 3),
                "mean_arch_progress": round(float(np.mean([r["arch_progress"] for r in c])), 4),
                "mean_first_recover_gen": round(float(np.mean(fgens)), 1) if fgens else None,
                "mean_n_on": round(float(np.mean([r["n_on"] for r in c])), 2),
                "mean_n_ref_kept": round(float(np.mean([r["n_ref_edges_kept"] for r in c])), 2),
                "mean_n_decoys_on": round(float(np.mean([r["n_decoys_on"] for r in c])), 2),
                "per_run": {f"s{r['seed']}_d{r['drop'][0]}": r["final_best"] for r in c},
            }
        per_W[W] = agg

    ws = sorted(per_W)
    vd = verdict_of(per_W, ws)

    def by_W(cond: str, field: str):
        # 条件子集时缺臂记 None, 不让 summary 构造在 KeyError 上翻车
        return {str(W): (per_W[W][cond][field] if cond in per_W[W] else None)
                for W in ws}

    summary = {
        "protocol": (
            "缺失边找回: 同起点(参考边集去 1 条关键边); "
            f"{len(conds)} 条件 {list(conds)} x 4 规模 x "
            f"{n_drops} 缺边 x {len(seeds)} 种子 x {gens} 代; pop {pop_size}; "
            "p_flip=2.1/W (翻转总量跨规模恒定); 条件共享 RNG 轨迹(逐种子配对); "
            "naive-arch=架构自由+参数钉在参考偏置(纯发现); param-only=架构冻结+参数自由(纯修复); "
            "directed=参考边集直装(天花板); sham=架构自由+参数钉住+适应度打乱(漂移地板)"
        ),
        "equivalence_check": eq,
        "reference_full_solve": full_solve,
        "dropped_edges": [[int(e) for e in d] for d in drops],
        "edge_criticality_top": crit[:5],
        "conds_run": list(conds),
        "per_W": {str(W): per_W[W] for W in ws},
        "recovery_rate_by_W": by_W("naive-arch", "recovery_rate"),
        "achieved_rate_by_W": by_W("naive-arch", "achieved_rate"),
        "sham_achieved_by_W": by_W("sham", "achieved_rate"),
        "param_only_recovery_by_W": by_W("param-only", "recovery_rate"),
        "param_only_score_by_W": by_W("param-only", "mean_final_best"),
        "directed_score_by_W": by_W("directed", "mean_final_best"),
        "wall_min": round((time.time() - t0) / 60, 1),
        **vd,
    }
    (out_dir / "scale_v2_verdict.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 曲线图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        style = {
            "naive-arch":  ("naive-arch(纯架构搜索)", "o-"),
            "naive-joint": ("naive-joint(双层联合)", "s--"),
            "param-only":  ("param-only(纯参数修复)", "^-"),
            "directed":    ("directed(定向通道/天花板)", "v:"),
            "sham":        ("sham(漂移地板)", "x-"),
        }
        for cond, (label, sty) in style.items():
            if cond not in per_W[ws[0]]:
                continue  # 条件子集: 只画在场的条件
            axes[0].plot(ws, [per_W[W][cond]["achieved_rate"] for W in ws], sty, label=label)
            axes[1].plot(ws, [per_W[W][cond]["mean_final_best"] for W in ws], sty, label=label)
        axes[0].set_title("Discovery: achievement rate (final >= 16/16)")
        axes[0].set_ylabel("achievement rate")
        axes[1].set_title("Final solved tasks")
        axes[1].set_ylabel("solved / 16")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("candidate edges W (log)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "scale_v2_curve.png", dpi=130)
        summary["plot"] = "scale_v2_curve.png"
        (out_dir / "scale_v2_verdict.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print("plot skip:", e)

    print()
    print(json.dumps({k: summary.get(k) for k in
                      ("recovery_rate_by_W", "achieved_rate_by_W", "sham_achieved_by_W",
                       "param_only_score_by_W", "directed_score_by_W",
                       "verdict", "checks", "achieved_last_over_first",
                       "arch_progress_arch_only_by_W", "arch_progress_joint_by_W",
                       "attention_gap_by_W", "wall_min")},
                     ensure_ascii=False, indent=1))
    print("WROTE", out_dir / "scale_v2_verdict.json")


if __name__ == "__main__":
    main()
