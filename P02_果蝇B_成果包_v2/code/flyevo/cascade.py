"""评估级联 (DGM 式): 用渐进抽样把评估算力集中到有希望的个体上。

原理
----
同一评估协议下, 个体适应度 = 它在 n_eps 个回合上统计量的均值。用**前 m 个回合**
(m < n_eps) 得到的均值是完整均值的**无偏估计** —— 只是方差更大。于是可以分级:

    L1  全体个体, 各跑 1 个回合                → 便宜的排名代理
    L2  只给前 50% 补到 2 个回合
    L3  只给前 25% 补到 n_eps 个回合           → 与完整协议逐位相同

两个关键性质 (实现与验证都依赖它们):

1. **回合是嵌套前缀**。第 m 级用的是 k = 0..m-1, 与完整协议的前 m 个回合完全一致。
   因此晋级到末级的个体, 其适应度与全量评估**逐位相同**, 不是近似值。
   前提: 调用方给出的 (个体, 回合) → 回合参数的映射必须与全量协议一致。
2. **不做任何缩放**。每个个体拿到的都是「它真正跑过的那些回合的均值」, 期望与完整
   协议一致。级联改变的只有测量精度, 而精度被有意分配给了排名靠前的个体。

代价 (以 全体 × n_eps 回合 为基准)
----------------------------------
    n_eps=2, 2 级 → 1 + 0.5×1          = 1.5 / 2   = 省 25%
    n_eps=4, 3 级 → 1 + 0.5 + 0.25×2   = 2.0 / 4   = 省 50%
    n_eps=8, 4 级 → 1 + 0.5 + 0.5 + 0.5 = 2.5 / 8  = 省 69%

所以级联真正的价值不是「省同一协议的算力」, 而是**让更稳的协议变得可负担**:
n_eps 从 2 提到 8 通常贵 4 倍, 级联后只贵 1.67 倍, 而测量噪声降到 1/2。

适用边界
--------
级联只应用于**非判决性**的多种子挂批 (P1.2 那类分布统计)。判决实验 (C7/C8 那类单点
结论) 应继续用全量评估 —— 级联下种群 mean_fit 更噪, 而它正是判决实验的读数。
`audit=True` 提供「级联是否改变了选择」的实测手段: 同一代里把未晋级个体的缺失回合
也补跑一遍, 与级联结果逐位对读。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

KEEP_FLOOR = 0.125
MIN_KEEP_FLOOR = 4


# ---------- 调度 ----------

def default_schedule(n_eps: int, max_levels: int = 3) -> list[tuple[float, int]]:
    """默认调度 [(保留比例, 累计回合数), ...]。末级累计回合数 == n_eps。

    回合目标按 1 → 2 → 4 → 8 … 翻倍 (与 DGM 的 10→50→200 同形), 保留比例逐级减半。
    """
    if n_eps <= 1:
        return [(1.0, int(n_eps))]

    eps_targets = [1]
    while len(eps_targets) < max(2, max_levels):
        nxt = min(int(n_eps), eps_targets[-1] * 2)
        if nxt <= eps_targets[-1]:
            break
        eps_targets.append(nxt)
    if eps_targets[-1] != int(n_eps):
        eps_targets.append(int(n_eps))

    keep = []
    kf = 1.0
    for _ in eps_targets:
        keep.append(kf)
        kf = max(KEEP_FLOOR, kf * 0.5)

    sched: list[tuple[float, int]] = []
    prev = 0
    for kf, e in zip(keep, eps_targets):
        if e == prev:
            continue
        sched.append((round(float(kf), 4), int(e)))
        prev = e
    return sched


def normalize_schedule(
    schedule: Sequence[tuple[float, int]] | None,
    n_eps: int,
    pop_size: int,
    min_keep: int,
) -> list[tuple[float, int]]:
    """把用户给的调度修正成合法形式: 首级全体、末级满回合、保留比例单调不增、保留数≥min_keep。

    注意首级只要求「全体参与且累计回合 ≥ 1」, 不强制恰好 1 个回合。
    强制 =1 会挡住一批**更保真**的调度: 首级全体各跑 2 个回合再筛, 成本比
    默认调度高但保真度好得多 (实测 n_eps=4 时精英一致率 0.33 → 见 bench 报告)。
    首级恰好 1 回合只该是 default_schedule 的选择, 不该是硬约束。
    """
    sched = [(float(a), int(b)) for a, b in schedule] if schedule else default_schedule(n_eps)
    sched = [(a, b) for a, b in sched if b > 0]
    if not sched:
        sched = default_schedule(n_eps)
    if sched[0][1] < 1:
        sched[0] = (sched[0][0], 1)
    # 首级永远是全体 (cascaded_evaluate 里 cohort = 全体), 这里把比例记成 1.0 保持一致
    sched[0] = (1.0, sched[0][1])
    sched[-1] = (sched[-1][0], int(n_eps))
    # 单调性: 保留比例不得上升
    for j in range(1, len(sched)):
        if sched[j][0] > sched[j - 1][0]:
            sched[j] = (sched[j - 1][0], sched[j][1])
    # 去掉累计回合数未增长的级
    out: list[tuple[float, int]] = []
    prev = 0
    for a, b in sched:
        if b <= prev:
            continue
        out.append((a, b))
        prev = b
    # 保留数下限: 末级至少留下 min_keep 个个体
    a, b = out[-1]
    floor_frac = min(1.0, max(a, min_keep / max(1, pop_size)))
    out[-1] = (float(floor_frac), b)
    return out


def schedule_cost(
    schedule: Sequence[tuple[float, int]],
    pop_size: int,
    min_keep: int = MIN_KEEP_FLOOR,
) -> int:
    """按调度实际要跑多少回合 (与 cascaded_evaluate 的 cohort 语义一致)。

    语义: `keep_frac` 是**进入该级的个体比例** —— 第 L 级取当前排名前
    keep_frac_L × pop 的个体, 给它们补跑到 eps_target_L 个回合。
    """
    total = 0
    prev = 0
    for li, (keep_frac, eps) in enumerate(schedule):
        if li == 0:
            n = pop_size
        else:
            n = min(pop_size, max(int(min_keep), int(round(keep_frac * pop_size))))
        total += n * (eps - prev)
        prev = eps
    return total


# ---------- 结果容器 ----------

@dataclass
class CascadeResult:
    stats_per_indiv: list[dict]     # 每个体的最终统计 (它跑过的回合的均值)
    fits: np.ndarray                # 适应度
    n_episodes_used: np.ndarray     # 每个体实际用了几个回合
    episodes_spent: int             # 本代实际回合数
    baseline_episodes: int          # 全量协议的回合数
    levels: list[dict]              # 每级记录
    promoted: np.ndarray            # 是否晋级到末级 (这些人的适应度与全量逐位相同)
    audit: dict | None = None       # audit=True 时的保真度读数

    @property
    def saving(self) -> float:
        if self.baseline_episodes <= 0:
            return 0.0
        return 1.0 - self.episodes_spent / self.baseline_episodes

    def log_fields(self) -> dict:
        d = {
            "episodes_spent": int(self.episodes_spent),
            "baseline_episodes": int(self.baseline_episodes),
            "cascade_saving": round(float(self.saving), 4),
            "n_promoted": int(self.promoted.sum()),
        }
        if self.audit:
            for k, v in self.audit.items():
                d[f"audit_{k}"] = v
        return d


# ---------- 主入口 ----------

def mean_stats(stats_list: Sequence[dict]) -> dict:
    keys = stats_list[0].keys()
    return {k: float(np.mean([s[k] for s in stats_list])) for k in keys}


def cascaded_evaluate(
    evaluate: Callable[[Sequence[tuple[int, int]]], list[dict]],
    fitness_fn: Callable[[dict], float],
    pop_size: int,
    n_eps: int,
    schedule: Sequence[tuple[float, int]] | None = None,
    min_keep: int = MIN_KEEP_FLOOR,
    audit: bool = False,
) -> CascadeResult:
    """按调度分级评估一个世代。

    参数
    ----
    evaluate(pairs) -> list[dict]
        pairs 是 (个体下标, 回合下标) 的列表; 返回值与 pairs 同序、同为原始 stats 字典。
        **同一个 (i, k) 必须与全量协议下那一回合完全一致** —— 嵌套前缀性质的前提。
    fitness_fn(stats) -> float
        与全量协议同一个适应度函数; 作用于「均值后的 stats」。
    audit
        True 时把未晋级个体缺的回合也补跑, 用于实测级联保真度 (成本 = 全量协议)。
    """
    pop_size = int(pop_size)
    n_eps = int(n_eps)
    sched = normalize_schedule(schedule, n_eps, pop_size, min_keep)

    stats_acc: list[list[dict]] = [[] for _ in range(pop_size)]
    episodes_spent = 0
    levels: list[dict] = []
    promoted = np.zeros(pop_size, dtype=bool)
    fits = np.zeros(pop_size)

    ranking = np.arange(pop_size)   # 上一级结束时的适应度排名
    prev_cohort = np.arange(pop_size)
    prev_eps = 0
    for li, (keep_frac, eps_target) in enumerate(sched):
        if eps_target <= prev_eps:
            continue
        # cohort = 进入本级的个体: 首级全体, 之后**只在上一级 cohort 内部**取前若干名。
        #
        # 这里必须嵌套 (L1 ⊇ L2 ⊇ L3 ⊇ ...), 不能拿全体重排。原因: 未晋级个体的
        # 适应度停留在它上一级的精度上 (1 回合 vs 4 回合), 混精度排名会让一个
        # 只跑了 2 回合的个体靠噪声挤进末级, 于是它拿到 6 个回合而不是 8 个 ——
        # 末级个体的适应度就不再与全量协议逐位相同, 整个「晋级者无偏」的性质塌掉。
        # (首版就是拿全体重排, 32 个个体里漏了 1 个, 由下面的硬校验抓出来。)
        if li == 0:
            cohort = np.arange(pop_size)
        else:
            n_keep = min(
                pop_size, max(int(min_keep), int(round(keep_frac * pop_size)))
            )
            n_keep = min(n_keep, len(prev_cohort))
            order = prev_cohort[np.argsort(fits[prev_cohort])[::-1]]
            cohort = order[:n_keep]

        pairs = [(int(i), k) for i in cohort for k in range(prev_eps, eps_target)]
        if pairs:
            out = evaluate(pairs)
            if len(out) != len(pairs):
                raise RuntimeError(
                    f"evaluate() 返回 {len(out)} 条, 期望 {len(pairs)} 条"
                )
            episodes_spent += len(pairs)
            for (i, _k), st in zip(pairs, out):
                stats_acc[i].append(st)
        prev_eps = eps_target

        # 只刷新本级的个体; 其余沿用上一级的估计 (仍是无偏的, 只是精度低)
        for i in cohort:
            fits[i] = float(fitness_fn(mean_stats(stats_acc[i])))
        ranking = np.argsort(fits)[::-1]
        prev_cohort = cohort

        is_last = li == len(sched) - 1
        if is_last:
            promoted[:] = False
            promoted[cohort] = True
        levels.append(
            {
                "level": li,
                "keep_frac": float(keep_frac),
                "eps_target": int(eps_target),
                "n_cohort": int(len(cohort)),
                "episodes_this_level": int(len(pairs)),
                "fit_mean_at_level": float(fits.mean()),
                "fit_max_at_level": float(fits.max()),
            }
        )

    stats_per_indiv = [
        mean_stats(stats_acc[i]) if stats_acc[i] else {}
        for i in range(pop_size)
    ]
    counts = np.array([len(s) for s in stats_acc], dtype=int)
    baseline = pop_size * n_eps

    # ---- 硬校验: 晋级者必须跑满 n_eps 个回合 ----
    # 这是「晋级者适应度与全量协议逐位相同」的直接前提。一旦 cohort 嵌套被破坏
    # (见上面注释里的混精度排名陷阱), 这里立刻失败, 而不是等到审计报告里
    # 出现一个说不清的 0.33 精英一致率。
    if promoted.any() and int(counts[promoted].min()) != n_eps:
        bad = np.flatnonzero(promoted & (counts != n_eps))
        raise AssertionError(
            f"级联不变量破坏: 晋级个体 {bad.tolist()} 只跑了 "
            f"{counts[bad].tolist()} 个回合, 应为 {n_eps}。"
            f"cohort 未嵌套 —— 检查 normalize_schedule / 调度是否让末级跨级取人。"
        )

    audit_stats = None
    if audit:
        missing = [(i, k) for i in range(pop_size) for k in range(counts[i], n_eps)]
        if missing:
            out = evaluate(missing)
            for (i, _k), st in zip(missing, out):
                stats_acc[i].append(st)
        fits_full = np.array(
            [float(fitness_fn(mean_stats(stats_acc[i]))) for i in range(pop_size)]
        )
        audit_stats = _audit_report(fits, fits_full, min_keep)
        audit_stats["episodes_full"] = int(baseline)
        audit_stats["episodes_cascade"] = int(episodes_spent)
        audit_stats["episodes_total_with_audit"] = int(episodes_spent + len(missing))

    return CascadeResult(
        stats_per_indiv=stats_per_indiv,
        fits=fits,
        n_episodes_used=counts,
        episodes_spent=int(episodes_spent),
        baseline_episodes=int(baseline),
        levels=levels,
        promoted=promoted,
        audit=audit_stats,
    )


# ---------- 保真度审计 ----------

def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩 (无 scipy 依赖)。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i: j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _audit_report(
    fits_cas: np.ndarray,
    fits_full: np.ndarray,
    elite: int,
    fracs: Sequence[float] = (0.05, 0.1, 0.25, 0.5),
) -> dict:
    """级联 vs 全量的保真度读数。

    spearman       秩相关 (1.0 = 排序完全一致)
    bias / mae     有符号偏差与平均绝对误差 (以全量为真值)
    top{K}_overlap 级联选出的前 K 名与全量前 K 名的重合率
    elite_exact    级联的精英集合是否与全量完全相同 (选择保真度的最硬读数)
    """
    n = len(fits_cas)
    rc, rf = _rankdata(fits_cas), _rankdata(fits_full)
    if rc.std() < 1e-12 or rf.std() < 1e-12:
        spearman = float("nan")
    else:
        spearman = float(np.corrcoef(rc, rf)[0, 1])

    out = {
        "spearman": round(spearman, 4) if spearman == spearman else None,
        "bias": round(float(np.mean(fits_cas - fits_full)), 4),
        "mae": round(float(np.mean(np.abs(fits_cas - fits_full))), 4),
        "fit_full_mean": round(float(np.mean(fits_full)), 4),
        # 全精度原值。接线自检 (bench_cascade) 必须比对**这两个** ——
        # 拿 round(x,4) 去比 1e-6 的容差, 量化误差最大 5e-5 会直接超容差,
        # 报出「接线不一致」的**假警报** (n_eps=8 的两份报告就是这样误报的)。
        "fit_full_mean_raw": float(np.mean(fits_full)),
        "fit_cas_mean_raw": float(np.mean(fits_cas)),
    }
    for f in fracs:
        k = max(1, int(round(f * n)))
        a = set(np.argsort(fits_cas)[::-1][:k].tolist())
        b = set(np.argsort(fits_full)[::-1][:k].tolist())
        out[f"top{int(round(f * 100))}_overlap"] = round(len(a & b) / k, 4)

    k = max(1, int(elite))
    a = set(np.argsort(fits_cas)[::-1][:k].tolist())
    b = set(np.argsort(fits_full)[::-1][:k].tolist())
    out["elite_exact"] = bool(a == b)
    out["elite_overlap"] = round(len(a & b) / k, 4)
    return out
