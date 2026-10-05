"""实验编排: 分阶段进化循环, 并行评估, 日志与报告."""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .agent import SensorimotorMapping, evaluate_episode
from .brain import BrainConfig, LIFBrain
from .evolution import GeneticAlgorithm, StructuralGA
from .cascade import cascaded_evaluate, normalize_schedule, schedule_cost
from . import neuro_evo
from . import structure_evo
from .stages import get_stages
from . import viz

ROOT = Path(__file__).resolve().parents[1]


def _mean_stats(stats_list: list[dict]) -> dict:
    keys = stats_list[0].keys()
    return {k: float(np.mean([st[k] for st in stats_list])) for k in keys}


# ---------- 并行 worker: 拓扑在 initializer 里建一次 ----------

_G: dict = {}


def _rebuild_worker(n_nodes, pre, post, sign, w0=None, nodes_records=None,
                    lesion_keep=None) -> None:
    cfg = BrainConfig()
    _G["brain"] = LIFBrain(n_nodes, pre, post, sign, cfg, w0=w0)
    _G["w0"] = None if w0 is None else np.asarray(w0)
    _G["mapping"] = SensorimotorMapping.from_nodes(pd.DataFrame(nodes_records))
    _G["universe_size"] = len(pre)
    if lesion_keep is not None:
        _G["lesion_keep"] = np.asarray(lesion_keep)
    else:
        _G.pop("lesion_keep", None)


def _worker_init(n_nodes, pre, post, sign, w0=None, nodes_records=None,
                 lesion_keep=None) -> None:
    _G["init"] = (n_nodes, pre, post, sign, nodes_records)
    _rebuild_worker(n_nodes, pre, post, sign, w0, nodes_records, lesion_keep)


def _apply_lesion(genome: np.ndarray) -> np.ndarray:
    """评估时把切断掩码乘到基因组上; 只作用于原始真实边前缀."""
    keep = _G.get("lesion_keep")
    if keep is None:
        return genome
    g = np.asarray(genome, dtype=float).copy()
    n = min(len(keep), len(g))
    g[:n] *= keep[:n]
    return g


def _eval_task(args):
    # 末位 *_extra 容错: _universe() 在接入递质基权重后多返回一个 w0 (P1.4),
    # 但 w0 是全程常量, 由 worker initializer 存进 _G, 不必逐任务传递。
    (genome, world_cfg_dict, env_seed, noise_seed, record_every,
     want_activity, pre, post, sign, *_extra) = args
    from . import world as W

    # 结构模式: 宇宙随代际生长, 载荷变长时 worker 用任务携带的宇宙重建大脑
    if _G.get("universe_size") != len(genome):
        n_nodes, nodes_records = _G["init"][0], _G["init"][4]
        _rebuild_worker(n_nodes, pre, post, sign, _G.get("w0"), nodes_records,
                        _G.get("lesion_keep"))

    cfg = W.WorldConfig(**world_cfg_dict)
    _G["brain"].set_weights(_apply_lesion(genome))
    stats, frames, activity = evaluate_episode(
        _G["brain"], _G["mapping"], cfg, env_seed, noise_seed,
        record_every=record_every, track_activity=want_activity,
    )
    return stats, frames if record_every else None, activity


# ---------- 数据加载与控制组 ----------

def load_connectome(control: str, topo_seed: int = 12345) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    nodes = pd.read_csv(ROOT / "data" / "flywire_subcircuit_nodes.csv")
    edges = pd.read_csv(ROOT / "data" / "flywire_subcircuit_edges.csv")

    # 初始权重: 真实突触计数的对数标度, 中位数归一到 1
    w_init = np.log1p(edges["syn_count"].to_numpy(float))
    w_init /= np.median(w_init)

    if control == "random-topo":
        # 保持每个前突神经元的出边数不变, 重配所有目标 -> 出度保持的随机图
        rng = np.random.default_rng(topo_seed)
        post = edges["post_idx"].to_numpy().copy()
        rng.shuffle(post)
        edges["post_idx"] = post
    elif control == "block-shuffle":
        # 群级块随机: 只在「后元所属群」内重抽具体目标 -> 群对边计数/各群总入度全保留,
        # 宏观嗅觉架构(ORN→PN→LH→DN)不变, 细粒度拓扑全随机.
        # 与 random-topo 对比可分离「宏观架构 vs 细微拓扑」各自对可进化性的贡献.
        rng = np.random.default_rng(topo_seed)
        group_of = nodes["group"].to_numpy()
        post = edges["post_idx"].to_numpy().copy()
        for g in pd.unique(group_of[post]):
            mask = group_of[post] == g
            members = np.flatnonzero(group_of == g)
            post[mask] = rng.choice(members, size=int(mask.sum()), replace=True)
        edges["post_idx"] = post
    elif control == "shuffled-weights":
        rng = np.random.default_rng(topo_seed)
        w_init = rng.permutation(w_init)
    elif control in ("none", "neutral"):
        pass
    return nodes, edges, w_init


# ---------- 主循环 ----------

def run_experiment(
    pop_size: int = 60,
    stage_gens: int = 40,
    duration: float = 20.0,
    seed: int = 1,
    control: str = "none",
    stages_filter: list[int] | None = None,
    workers: int = 1,
    n_eps: int = 2,
    structure: bool = False,
    p_add: float = 0.25,
    p_del: float = 0.25,
    p_dup: float = 0.03,
    out_dir: str | None = None,
    lesion_frac: float = 0.0,
    lesion_pathway: str = "ORN_PN",
    lesion_seed: int = 0,
    new_edge_w: float = 0.8,
    structure_elite: int = 1,
    directed_add: bool = True,
    mut_sigma: float = 0.15,
    weight_crossover: bool = True,
    candidate_edges: list | None = None,
    topo_seed: int = 12345,
    cascade: bool = False,
    cascade_schedule: list[tuple[float, int]] | None = None,
    cascade_audit: bool = False,
    cascade_min_keep: int = 4,
    nt_policy: str = "gaba-only",
    edge_weight: str = "unit",
    kappa_price: float = 0.0,
    kappa_hz: float = 1.0,
    offers_path: str | None = None,
) -> Path:
    # G6.2.0: 认知成本 κ 耦合进适应度 (活跃神经元 × 代谢单价), 默认 0 = 历史 run 逐位不变
    if kappa_price > 0 and cascade:
        raise ValueError("κ 耦合与级联评估互斥 (判决实验禁用级联)")
    # neutral = 中性对照: 拓扑/环境/评估全部相同, 但适应度随机打乱后再传给 GA
    # -> 度量"没有选择时, 神经元层面纯靠突变-漂移能走多远"
    neutral = control == "neutral"
    # spike-in 受控注入: 预生成候选边提议表 (CSV), 两臂读同一张表 (论文 4.3 预注册)
    offers = None
    offers_sha = None
    if offers_path:
        if not structure:
            raise ValueError("offers_path 只支持 structure 模式 —— 拒绝静默忽略")
        import csv as _csv
        import hashlib as _hashlib

        _raw = Path(offers_path).read_bytes()
        offers_sha = _hashlib.sha256(_raw).hexdigest()
        offers = []
        with Path(offers_path).open(newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                offers.append((
                    int(r["gen"]), int(r["child"]),
                    {"do_del": r["do_del"] == "1", "do_add": r["do_add"] == "1",
                     "pre": int(r["pre"]), "post": int(r["post"]),
                     "do_dup": r["do_dup"] == "1",
                     "dup_src": int(r["dup_src"]), "dup_dst": int(r["dup_dst"])},
                ))
        if not offers:
            raise ValueError(f"提议表为空: {offers_path} —— 拒绝空表静默跑批")
    # structure = 结构突变模式: 加边/剪边/神经元重复 (StructuralGA)
    out = Path(out_dir) if out_dir else ROOT / "runs" / f"exp_{control}_s{seed}_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)

    nodes, edges, w_init = load_connectome(control, topo_seed)
    stages = get_stages(stages_filter)
    pre_idx = edges["pre_idx"].to_numpy()
    post_idx = edges["post_idx"].to_numpy()
    # 递质符号基质 (P1.4): 默认 (gaba-only, unit) 与 v1 逐位一致。
    # policy 决定 sign; weight_mode 决定固定基权重 w0 (syn 口径下逐对求和 = N_exc − N_inh)
    from .nt_sign import load_substrate

    sign_arr, w0_arr, nt_meta = load_substrate(nt_policy, edge_weight)
    if structure:
        w0_arr = None   # 结构模式下宇宙随代生长, 基权重口径不适用
    group_ids = nodes["group"].map({"ORN": 0, "PN": 1, "LH": 2, "DN": 3}).to_numpy()

    lesion_keep = None
    lesion_info = None
    if lesion_frac > 0:
        from .lesion import LesionSpec, lesion_mask, lesion_summary

        lesion_keep = lesion_mask(
            pre_idx, post_idx, nodes,
            LesionSpec(frac=lesion_frac, pathway=lesion_pathway, seed=lesion_seed),
        )
        lesion_info = lesion_summary(lesion_keep)
        # 初始权重也乘 keep, 让「被切断」从第 0 代就是硬约束
        w_init = w_init * lesion_keep.astype(float)

    if structure:
        # 群 id: ORN=0 PN=1 LH=2 DN=3 — 与 group_ids 映射一致
        # 旁路白名单: 切断主感觉通路后, 结构搜索应偏向绕行通路
        flows = None
        if directed_add:
            flows = ((0, 2), (0, 3), (1, 3), (2, 3))  # ORN→LH, ORN→DN, PN→DN, LH→DN
        # 中性对照用独立 GA 突变流, 避免与真实组共享同一突变序列造成「假共有边」
        ga_seed = seed + 10_000 if neutral else seed
        ga = StructuralGA(
            pre_idx, post_idx, sign_arr, w_init, n_nodes=len(nodes),
            group=group_ids, pop_size=pop_size, seed=ga_seed,
            p_add=p_add, p_del=p_del, p_dup=p_dup,
            new_edge_w=new_edge_w, elite=structure_elite,
            allowed_flows=flows,
            # candidate_edges 非空时加边/副本都限于显式候选池 (A2b), 群对白名单失效
            allowed_edges=candidate_edges,
            mut_sigma=mut_sigma,
            weight_crossover=weight_crossover,
            offer_table=offers,
        )
    else:
        ga_seed = seed + 10_000 if neutral else seed
        ga = GeneticAlgorithm(w_init, pop_size=pop_size, seed=ga_seed,
                              mut_sigma=mut_sigma)

    def _mean_eff() -> np.ndarray:
        return ga.mean_effective() if structure else ga.genomes.mean(axis=0)

    def _payload(i: int) -> np.ndarray:
        return ga.payload(i) if structure else ga.genomes[i]

    def _uni_size() -> int:
        return int(ga.size) if structure else int(len(pre_idx))

    (out / "config.json").write_text(
        json.dumps(
            {
                    "pop_size": pop_size, "stage_gens": stage_gens, "duration": duration,
                    "seed": seed, "control": control, "structure": structure,
                    "topo_seed": topo_seed,
                    "ga_seed": (seed + 10_000) if neutral else seed,
                "independent_neutral_ga": bool(neutral),
                "p_add": p_add, "p_del": p_del, "p_dup": p_dup,
                "stages": [s["idx"] for s in stages],
                "n_nodes": len(nodes), "n_edges": len(edges),
                "lesion": lesion_info,
                "new_edge_w": new_edge_w if structure else None,
                "structure_elite": structure_elite if structure else None,
                "directed_add": directed_add if structure else None,
                "mut_sigma": mut_sigma,
                "weight_crossover": weight_crossover if structure else None,
                "n_candidate_edges": len(candidate_edges) if (structure and candidate_edges) else None,
                "shared_offer_supply": bool(offers_path),
                "offers_path": offers_path,
                "offers_sha256": offers_sha,
                "n_offers": len(offers) if offers else None,
                "ga_seed": seed + 10_000 if neutral else seed,
                "neutral_independent_ga": bool(neutral),
                "cascade": bool(cascade),
                "cascade_schedule": cascade_schedule if cascade else None,
                "cascade_audit": bool(cascade_audit) if cascade else None,
                "cascade_min_keep": cascade_min_keep if cascade else None,
                "cascade_effective_schedule": (
                    normalize_schedule(cascade_schedule, n_eps, pop_size, cascade_min_keep)
                    if cascade else None
                ),
                "nt_policy": nt_policy,
                "edge_weight": edge_weight,
                "nt_substrate": nt_meta,
                "kappa_price": kappa_price,
                "kappa_hz": kappa_hz if kappa_price > 0 else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    world_dicts = [{k: v for k, v in s["world"].__dict__.items()} for s in stages]
    for wd in world_dicts:
        wd["duration"] = duration

    def _universe():
        if structure:
            return (ga.U_pre[: ga.size], ga.U_post[: ga.size], ga.U_sign[: ga.size], None)
        return (pre_idx, post_idx, sign_arr, w0_arr)

    def _build_pool():
        return ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(len(nodes), *_universe(), nodes.to_dict("records"), lesion_keep),
        )

    # 主进程的串行路径也要有 _G 初始状态 (结构模式下宇宙会生长)
    _worker_init(len(nodes), *_universe(), nodes.to_dict("records"), lesion_keep)
    rows: list[dict] = []
    stage_summaries = []
    freqs: list[np.ndarray] = []   # 结构模式: 每代边存在频率快照
    struct_rows: list[dict] = []
    cascade_runs: list[dict] = []  # 级联模式: 每代的成本与保真度记录
    global_gen = 0

    for stage in stages:
        executor = _build_pool() if workers > 1 else None
        try:
            wd = world_dicts[stages.index(stage)]
            wd["duration"] = duration
            stage_mean_start = _mean_eff().copy()
            snaps = [stage_mean_start]  # 每代种群平均基因组快照 -> 神经元代际轨迹
            fit_hist = []

            pbar = tqdm(
                range(stage_gens),
                desc=f"S{stage['idx']} {stage['name']}",
                unit="gen",
            )
            for gen in pbar:
                # 每个个体评估 n_eps 个环境取均值, 压低世代间环境布局差异的噪声
                env_seed = seed * 1_000_000 + stage["idx"] * 10_000 + gen

                def _eval_pairs(pairs, _env=env_seed, _wd=wd):
                    """按 (个体, 回合) 对评估。同一 (i,k) 的回合参数与全量协议完全一致,
                    这是级联「嵌套前缀」性质成立的前提。
                    κ 模式下额外返回每回合的 activity 数组 (主进程数活跃神经元)。"""
                    tasks = [
                        (
                            _payload(i), _wd, _env + k * 77_777,
                            _env * 97 + i + k * 13, 0, kappa_price > 0,
                            *_universe(),
                        )
                        for (i, k) in pairs
                    ]
                    if executor:
                        res = list(executor.map(_eval_task, tasks, chunksize=2))
                    else:
                        res = [_eval_task(t) for t in tasks]
                    if kappa_price > 0:
                        return [r[0] for r in res], [r[2] for r in res]
                    return [r[0] for r in res]

                if cascade:
                    cres = cascaded_evaluate(
                        _eval_pairs, stage["fitness"], pop_size, n_eps,
                        schedule=cascade_schedule, audit=cascade_audit,
                        min_keep=cascade_min_keep,
                    )
                    stats, fits = cres.stats_per_indiv, cres.fits
                    cascade_fields = cres.log_fields()
                    cascade_runs.append({
                        "stage": stage["idx"], "gen": gen,
                        "episodes_spent": int(cres.episodes_spent),
                        "baseline_episodes": int(cres.baseline_episodes),
                        "saving": round(float(cres.saving), 4),
                        "n_promoted": int(cres.promoted.sum()),
                        "levels": cres.levels,
                        "audit": cres.audit,
                    })
                else:
                    # 保持原有任务顺序 (k 外 i 内), 让历史 run 逐位可复现
                    pairs = [(i, k) for k in range(n_eps) for i in range(pop_size)]
                    if kappa_price > 0:
                        raw, acts = _eval_pairs(pairs)
                        active = np.array([
                            float((acts[k * pop_size + i] > kappa_hz).sum())
                            for (i, k) in pairs
                        ]).reshape(n_eps, pop_size).mean(axis=0)
                    else:
                        raw = _eval_pairs(pairs)
                    stats = [
                        _mean_stats([raw[k * pop_size + i] for k in range(n_eps)])
                        for i in range(pop_size)
                    ]
                    fits = np.array([stage["fitness"](st) for st in stats])
                    if kappa_price > 0:
                        # G6.2.0: 适应度扣减认知成本 κ = 代谢单价 × 活跃神经元数
                        fits = fits - kappa_price * active
                    cascade_fields = {}

                fit_hist.append(fits)

                row = {
                    "stage": stage["idx"],
                    "gen": gen,
                    "best_fit": float(fits.max()),
                    "mean_fit": float(fits.mean()),
                    "std_fit": float(fits.std()),
                    "mean_food": float(np.mean([s["food"] for s in stats])),
                    "mean_toxin_s": float(np.mean([s["toxin_s"] for s in stats])),
                    "mean_alive_frac": float(np.mean([s["alive_frac"] for s in stats])),
                    "diversity": ga.diversity(),
                    **cascade_fields,
                }
                if kappa_price > 0:
                    row["mean_active"] = float(active.mean())
                    row["mean_cost"] = float((kappa_price * active).mean())
                rows.append(row)
                snaps.append(_mean_eff().copy())
                global_gen += 1
                if structure:
                    freqs.append(ga.edge_freq())
                    ev = ga.events_hist[-1] if ga.events_hist else {"add": 0, "del": 0, "dup": 0}
                    struct_rows.append({
                        "stage": stage["idx"], "gen": gen,
                        "add": ev["add"], "del": ev["del"], "dup": ev["dup"],
                        "universe": ga.size,
                        "mean_present": float(ga.mask[:, : ga.size].mean()),
                    })
                pbar.set_postfix(
                    best=f"{row['best_fit']:.1f}",
                    mean=f"{row['mean_fit']:.1f}",
                    food=f"{row['mean_food']:.1f}",
                )
                ga_r = np.random.default_rng(seed * 7 + stage["idx"] * 101 + gen)
                fit_use = ga_r.permutation(fits) if neutral else fits
                if structure:
                    ga.evolve(fit_use, global_gen)
                else:
                    ga.evolve(fit_use)

            # ---- 阶段收尾: 回放最优个体 + 神经元层进化分析 ----
            fits_last = fit_hist[-1]
            best_genome = (
                ga.best_payload(fits_last) if structure else ga.best(fits_last)[1]
            )
            # 阶段起点快照按当前宇宙补零 (新边在起点"不存在"= 权重 0)
            start_full = np.zeros(_uni_size())
            start_full[: len(stage_mean_start)] = stage_mean_start
            replay_seed = seed * 1_000_000 + stage["idx"] * 10_000 + 999
            task_end = (best_genome, wd, replay_seed, replay_seed, 6, True, *_universe())
            task_start = (start_full, wd, replay_seed, replay_seed + 1, 0, True, *_universe())
            if executor:
                stats_best, frames, hz_end = executor.submit(_eval_task, task_end).result()
                _, _, hz_start = executor.submit(_eval_task, task_start).result()
            else:
                stats_best, frames, hz_end = _eval_task(task_end)
                _, _, hz_start = _eval_task(task_start)
            if frames:
                viz.render_replay(frames, stage["world"], out / f"replay_stage{stage['idx']}.gif")

            stage_mean_end = _mean_eff()
            snaps_mat = np.zeros((len(snaps), _uni_size()))
            for si, sv in enumerate(snaps):
                snaps_mat[si, : len(sv)] = sv
            np.save(out / f"genome_trace_stage{stage['idx']}.npy", snaps_mat)
            pre_u, post_u, sign_u, _w0_u = _universe()
            origin_u = (
                ga.U_origin[: ga.size] if structure
                else np.zeros(_uni_size(), dtype=np.int8)
            )
            added_gen_u = (
                ga.U_added_gen[: ga.size] if structure
                else np.full(_uni_size(), -1, dtype=np.int32)
            )
            wdiff = viz.weight_diff_report(
                nodes, pre_u, post_u, sign_u, start_full, stage_mean_end,
                out / f"weight_diff_stage{stage['idx']}.csv",
                origin=origin_u, added_gen=added_gen_u, syn_table=edges,
            )
            neuro_tab = neuro_evo.neuron_evo_table(
                nodes, pre_u, post_u, sign_u,
                start_full, stage_mean_end, hz_start, hz_end,
            )
            neuro_tab.to_csv(out / f"neuron_evo_stage{stage['idx']}.csv", index=False)
            neuro_sum = neuro_evo.summarize_neuron_evo(neuro_tab)
            viz.plot_neuron_influence(
                neuro_tab, stage["world"], out / f"neuron_influence_stage{stage['idx']}.png"
            )
            viz.plot_neuron_traces(
                neuro_tab, snaps_mat, pre_u, post_u, sign_u,
                out / f"neuron_trace_stage{stage['idx']}.png",
            )
            stage_summaries.append(
                {
                    "stage": stage["idx"],
                    "name": stage["name"],
                    "gens": stage_gens,
                    "final_best": float(fits_last.max()),
                    "final_mean": float(fits_last.mean()),
                    "best_replay": stats_best,
                    "neuro": neuro_sum,
                }
            )
            gen_log = pd.DataFrame(rows)
            viz.plot_fitness(gen_log, out / "fitness.png")
            gen_log.to_csv(out / "gen_log.csv", index=False)
        finally:
            if executor:
                executor.shutdown()

    gen_log = pd.DataFrame(rows)
    viz.plot_fitness(gen_log, out / "fitness.png")
    gen_log.to_csv(out / "gen_log.csv", index=False)

    if cascade and cascade_runs:
        write_cascade_summary(out, cascade_runs, pop_size, n_eps,
                              cascade_schedule, cascade_min_keep)

    if structure and freqs:
        registry = pd.DataFrame({
            "pre_idx": ga.U_pre[: ga.size].astype(int),
            "post_idx": ga.U_post[: ga.size].astype(int),
            "sign": ga.U_sign[: ga.size],
            "origin": ga.U_origin[: ga.size],
            "added_gen": ga.U_added_gen[: ga.size],
            "final_freq": ga.edge_freq(),
        })
        registry.to_csv(out / "edge_registry.csv", index=False)
        gens_a, eids_a, vals_a = structure_evo.freq_snapshot_to_long(freqs)
        np.savez_compressed(out / "edge_freq.npz", gens=gens_a, eids=eids_a, vals=vals_a)
        pd.DataFrame(ga.add_log, columns=["gen", "pre", "post", "edge_id", "reactivated"]).to_csv(
            out / "add_log.csv", index=False
        )
        dup_df = pd.DataFrame(ga.dup_log, columns=["gen", "src", "dst", "n_copied", "n_old_cut"])
        dup_df.to_csv(out / "dup_log.csv", index=False)
        pd.DataFrame(struct_rows).to_csv(out / "structure_log.csv", index=False)

        freq_matrix = np.full((len(freqs), ga.size), np.nan)
        for gi, f in enumerate(freqs):
            freq_matrix[gi, : len(f)] = f
        fix = structure_evo.analyze_fixation(registry, freq_matrix, list(range(len(freqs))))
        dup_surv = structure_evo.duplication_survival(dup_df, registry, freq_matrix)
        viz.plot_edge_fixation(registry, freq_matrix, fix, nodes, out / "edge_fixation.png")
        structure_evo.write_report(out, registry, fix, dup_surv, struct_rows, nodes)

    write_report(out, control, seed, stages, stage_summaries, gen_log)
    return out


# ---------- 级联汇总 ----------

def write_cascade_summary(
    out: Path,
    runs: list[dict],
    pop_size: int,
    n_eps: int,
    schedule: list[tuple[float, int]] | None,
    min_keep: int,
) -> None:
    """级联评估的成本与保真度汇总 —— P1.3 的验收产物。

    成本: 预测值 (schedule_cost) 与实际值 (逐代累计) 对读, 两者必须一致。
    保真度: 只有 cascade_audit=True 时才有 —— 同一代内把未晋级个体缺的回合补跑,
    用全量结果当真值, 检查级联是否改变了选择。
    """
    eff = normalize_schedule(schedule, n_eps, pop_size, min_keep)
    pred = schedule_cost(eff, pop_size, min_keep)
    base = pop_size * n_eps
    spent = int(sum(r["episodes_spent"] for r in runs))
    baseline = int(sum(r["baseline_episodes"] for r in runs))
    audited = [r for r in runs if r.get("audit")]

    summary: dict = {
        "pop_size": pop_size,
        "n_eps": n_eps,
        "schedule": eff,
        "episodes_per_gen_predicted": pred,
        "episodes_per_gen_baseline": base,
        "saving_predicted": round(1 - pred / base, 4) if base else 0.0,
        "gens": len(runs),
        "episodes_spent_total": spent,
        "episodes_baseline_total": baseline,
        "saving_actual": round(1 - spent / baseline, 4) if baseline else 0.0,
        "n_gens_audited": len(audited),
    }

    if audited:
        def _mean_of(key: str) -> float | None:
            vals = [a["audit"][key] for a in audited if a["audit"].get(key) is not None]
            return round(float(np.mean(vals)), 4) if vals else None

        mae = _mean_of("mae")
        scale = float(np.mean([a["audit"]["fit_full_mean"] for a in audited]))
        summary["audit"] = {
            "spearman_mean": _mean_of("spearman"),
            "bias_mean": _mean_of("bias"),
            "mae_mean": mae,
            "mae_rel": (round(mae / scale, 4) if mae is not None and abs(scale) > 1e-9 else None),
            "elite_exact_rate": round(
                float(np.mean([1.0 if a["audit"]["elite_exact"] else 0.0 for a in audited])), 4
            ),
            "top5_overlap_mean": _mean_of("top5_overlap"),
            "top10_overlap_mean": _mean_of("top10_overlap"),
            "top25_overlap_mean": _mean_of("top25_overlap"),
            "top50_overlap_mean": _mean_of("top50_overlap"),
        }

    (out / "cascade_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# 评估级联 · 成本与保真度汇总",
        "",
        f"- 种群 {pop_size} | n_eps {n_eps} | 调度 {eff}",
        f"- 每代回合数: 级联 **{pred}** vs 全量 {base} "
        f"→ 预测省 **{summary['saving_predicted'] * 100:.1f}%**",
        f"- 实跑 {len(runs)} 代: 级联 {spent} 回合 vs 全量 {baseline} "
        f"→ 实际省 **{summary['saving_actual'] * 100:.1f}%**",
        "",
    ]
    if audited:
        a = summary["audit"]
        lines += [
            "## 保真度审计（同一代内补跑缺失回合，与级联结果对读）",
            "",
            f"- 秩相关 (Spearman) 均值: **{a['spearman_mean']}**",
            f"- 精英集合完全一致率: **{a['elite_exact_rate']}**",
            f"- top5 / top10 / top25 / top50 重合率: {a['top5_overlap_mean']} / "
            f"{a['top10_overlap_mean']} / {a['top25_overlap_mean']} / {a['top50_overlap_mean']}",
            f"- 有符号偏差 {a['bias_mean']} | 平均绝对误差 {a['mae_mean']} "
            f"(相对 {a['mae_rel']})",
            "",
            "> 判读: 秩相关与 top-K 重合率衡量「级联是否改变了选择」。",
            "> 若 top-K 重合率明显低于 1 或精英集合不一致，该协议下不应使用级联。",
            "",
        ]
    (out / "cascade_summary.md").write_text("\n".join(lines), encoding="utf-8")

    tail = ""
    if audited:
        tail = (f" | 秩相关 {summary['audit']['spearman_mean']}"
                f" | 精英一致率 {summary['audit']['elite_exact_rate']}")
    print(f"[cascade] 每代回合 {pred}/{base} (省 {summary['saving_predicted'] * 100:.1f}%){tail}")


# ---------- 报告 ----------

def write_report(
    out: Path, control: str, seed: int, stages: list,
    summaries: list[dict], gen_log: pd.DataFrame,
) -> None:
    lines = [
        "# Fly-Evo 实验报告",
        "",
        f"- 控制组: **{control}** | 种子: {seed}",
        f"- 神经元: {len(pd.read_csv(ROOT / 'data' / 'flywire_subcircuit_nodes.csv'))} "
        "(真实 FlyWire 嗅觉通路子回路)",
        "",
    ]
    for s, summ in zip(stages, summaries):
        lines += [
            f"## 阶段 {s['idx']}: {s['name']}",
            f"- {s['desc']}",
            f"- 收尾适应度: best={summ['final_best']:.1f}, mean={summ['final_mean']:.1f}",
            f"- 最优个体回放: food={summ['best_replay']['food']}, "
            f"toxin={summ['best_replay']['toxin_s']}s, alive={summ['best_replay']['alive_frac']*100:.0f}%",
            "",
        ]
        neuro = summ.get("neuro")
        if neuro:
            lines += [
                f"- **神经元层进化**: {neuro['n_changed_2sigma']}/{neuro['n_neurons']} "
                f"个神经元的下行影响力显著改变 ({neuro['frac_changed']*100:.0f}%)",
                f"  - 功能征用(沉默→活跃): {neuro.get('recruited', '-')} 个 | "
                f"功能沉默(活跃→沉默): {neuro.get('silenced', '-')} 个",
                "  - 影响力漂移 top5: " + "; ".join(
                    f"{g}/{t}({d:+.1f})" for g, t, d in neuro["top_movers"][:5]
                ),
                "  - 分群净漂移: "
                + ", ".join(f"{g}: {v:+.2f}" for g, v in neuro["group_shift"].items()),
                "",
            ]
        d = gen_log[gen_log["stage"] == s["idx"]]
        first5 = d.head(5)["mean_food"].mean()
        last5 = d.tail(5)["mean_food"].mean()
        lines.append(f"- 觅食均值: 前5代 {first5:.1f} -> 后5代 {last5:.1f}")

    # 类演化现象自动判定
    lines += ["", "## 涌现现象检视", ""]
    for s in stages:
        d = gen_log[gen_log["stage"] == s["idx"]]
        if d.empty:
            continue
        if s["idx"] == 1:
            gain = d.tail(5)["mean_food"].mean() - d.head(5)["mean_food"].mean()
            lines.append(f"- **趋化适应**: S1 后5代觅食比前5代多 {gain:+.1f} 颗"
                         + (" (出现适应)" if gain > 0.5 else " (未见明显适应)"))
        if s["idx"] == 2:
            t0, t1 = d.head(5)["mean_toxin_s"].mean(), d.tail(5)["mean_toxin_s"].mean()
            lines.append(f"- **毒物回避**: 毒物暴露 {t0:.2f}s -> {t1:.2f}s"
                         + (" (出现回避)" if t1 < t0 * 0.8 else ""))
        if s["idx"] == 3:
            a0 = d.head(5)["mean_alive_frac"].mean()
            a1 = d.tail(5)["mean_alive_frac"].mean()
            lines.append(f"- **逃逸进化**: 存活率 {a0*100:.0f}% -> {a1*100:.0f}%"
                         + (" (出现逃逸适应)" if a1 > a0 + 0.05 else ""))
        if s["idx"] == 4:
            a_min = d["mean_alive_frac"].min()
            lines.append(f"- **剧变响应**: 最低世代存活率 {a_min*100:.0f}%"
                         + (" (部分灭绝!)" if a_min < 0.4 else " (种群稳定渡过)"))
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
