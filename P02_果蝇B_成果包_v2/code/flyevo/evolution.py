"""遗传算法: 基因组 = 全部突触权重(乘性标度), 初始来自真实突触计数.

StructuralGA 在此之上支持结构突变: 边宇宙 + 存在掩码表示, 加边/剪边/
神经元重复三算子, 边的存在频率直接类比等位基因频率.
"""
from __future__ import annotations

import numpy as np


class GeneticAlgorithm:
    def __init__(
        self,
        w_init: np.ndarray,
        pop_size: int = 60,
        elite: int = 2,
        tournament: int = 3,
        mut_sigma: float = 0.15,
        w_min: float = 0.01,
        w_max: float = 15.0,
        seed: int = 0,
    ):
        self.w_init = w_init
        self.pop_size = pop_size
        self.elite = elite
        self.tournament = tournament
        self.mut_sigma = mut_sigma
        self.w_min = w_min
        self.w_max = w_max
        self.rng = np.random.default_rng(seed)
        # 个体初始 = 真实权重 * 对数正态扰动, 让种群有起点差异
        self.genomes = w_init[None, :] * self.rng.lognormal(
            0.0, 0.10, size=(pop_size, len(w_init))
        )
        self.genomes = np.clip(self.genomes, w_min, w_max)

    def diversity(self) -> float:
        """基因组平均标准差(跨个体), 归一化到初值尺度."""
        denom = np.where(np.abs(self.w_init) > 1e-12, self.w_init, 1.0)
        return float((self.genomes.std(axis=0) / denom).mean())

    def evolve(self, fitness: np.ndarray) -> None:
        """按适应度选择/交叉/变异, 生成下一代."""
        rng = self.rng
        order = np.argsort(fitness)[::-1]
        elites = self.genomes[order[: self.elite]].copy()

        def pick() -> int:
            cand = rng.integers(0, self.pop_size, size=self.tournament)
            return int(cand[np.argmax(fitness[cand])])

        children = []
        while len(children) < self.pop_size - self.elite:
            pa, pb = pick(), pick()
            # 均匀交叉
            mask = rng.random(len(self.w_init)) < 0.5
            child = np.where(mask, self.genomes[pa], self.genomes[pb])
            # 高斯突变
            child = child * rng.normal(1.0, self.mut_sigma, size=child.shape)
            children.append(np.clip(child, self.w_min, self.w_max))

        self.genomes = np.vstack([elites, np.array(children)])

    def best(self, fitness: np.ndarray) -> tuple[int, np.ndarray]:
        i = int(np.argmax(fitness))
        return i, self.genomes[i]


class StructuralGA:
    """可变结构基因组: 共享边宇宙 U + 每个体一个存在掩码.

    U 里的每条边有 origin: real(原始连接, 可被遗传性地剪除但永远留在宇宙里)
    或 added(进化中发明的新连接, 频率归零即被回收). 边的存在频率在种群层面
    直接类比等位基因频率, 是结构固定/灭绝分析的原始数据.

    评估端不变: 传给仿真的就是有效权重 w*mask(不存在的边权重为 0).
    """

    ORIGIN_REAL = 0
    ORIGIN_ADDED = 1

    def __init__(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        sign: np.ndarray,
        w_init: np.ndarray,
        n_nodes: int,
        group: np.ndarray,
        pop_size: int = 60,
        elite: int = 1,
        tournament: int = 3,
        mut_sigma: float = 0.15,
        w_min: float = 0.01,
        w_max: float = 15.0,
        p_add: float = 0.25,
        p_del: float = 0.25,
        p_dup: float = 0.03,
        new_edge_w: float = 0.8,
        allowed_flows: tuple | None = None,
        allowed_edges: list | None = None,
        weight_crossover: bool = True,
        offer_table: list | None = None,
        credit_gain: float = 0.0,
        credit_mode: str = "symmetric",
        credit_ema_decay: float = 0.8,
        bundle_cap: int = 0,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.elite = elite
        self.tournament = tournament
        self.mut_sigma = mut_sigma
        self.w_min = w_min
        self.w_max = w_max
        self.p_add = p_add
        self.p_del = p_del
        self.p_dup = p_dup
        self.new_edge_w = new_edge_w
        # P-A 信用分配 (R7 §10): >0 时对"新增边"启用反事实信用加权遗传.
        # 0 = 关闭, 行为与历史完全一致.
        self.credit_gain = credit_gain
        # 信用使用方式 (R7 §10.5 修正):
        #   "symmetric"   正信用奖励 / 负信用惩罚 (对称) —— 初版, 实测把新边携带者数
        #                 从 ~5 压到 ~0.3, 结构探索整体熄火且 target/decoy 无区分.
        #   "reward_only" 只奖励正信用, 不惩罚 —— 保护"早期无回报"的探索变异.
        #   "reward_ema"  同 reward_only, 且信用跨代指数累积以降低估计噪声.
        self.credit_mode = credit_mode
        self.credit_ema_decay = float(credit_ema_decay)
        self._credit_ema: np.ndarray | None = None
        # P-B 携带数压缩 (R7 §10.5): >0 时每个子代携带的新增边超过该值则随机
        # 保留 cap 条、其余清零 —— 评估粒度=变异粒度的直接杠杆。0 = 关闭且
        # 不消耗随机数, 历史 run 逐位不变。
        self.bundle_cap = int(bundle_cap)
        # 基线边数 = 原始连接组边数; 信用分配只作用于 idx >= n_base 的新增边.
        self.n_base = int(len(np.asarray(w_init)))
        self.n_nodes = n_nodes
        self.group = np.asarray(group)
        # 白名单加边: ((pre_group, post_group), ...) 群 id 见 runner 的映射
        # None = 全空间均匀随机 (旧默认)
        self.allowed_flows = allowed_flows
        # 候选边列表: [(pre, post), ...] 显式边池 (A2b 搜索空间缩减)。
        # 非空时 _op_add 只从池里均匀采样, _op_dup 也只放行落在池内的副本边,
        # 否则重复算子会绕开缩减; None = 不设限 (旧默认)。
        self.allowed_edges = (
            [(int(p), int(q)) for p, q in allowed_edges] if allowed_edges else None
        )
        self._allowed_edge_set = set(self.allowed_edges) if self.allowed_edges else None
        # A2a 权重冻结选型 (设计文档注明的第二种方案, 落笔即冻结): mut_sigma=0 只
        # 冻住高斯变异, 交叉仍会拼贴双亲权重; weight_crossover=False 让子代权重整行
        # 继承单个父代, 权重层才真正零变化。结构掩码通道 (交叉/加/剪/重复) 照常。
        self.weight_crossover = weight_crossover
        self._group_nodes = {
            int(g): np.flatnonzero(self.group == g) for g in np.unique(self.group)
        }
        # spike-in 受控注入: 预先生成的候选边提议表 (论文 4.3 预注册设计)。
        # 键 (gen, child) -> 供给侧决策 (算子开火/加边提议/重复源目标);
        # 两臂读同一张表, 只差适应度是否参与选择 —— 剪边目标与锦标赛仍用 self.rng。
        # None = 旧默认 (self.rng 自带供给), 历史 run 逐位不变。
        self.offer_table = (
            {(int(g), int(c)): o for g, c, o in offer_table} if offer_table else None
        )

        # ---- 边宇宙: 预留 1024 槽位给进化中新增的边 ----
        self.size = len(pre)
        self._capacity = self.size + 1024
        pad = self._capacity - self.size
        self.U_pre = np.pad(pre.astype(np.int64), (0, pad))
        self.U_post = np.pad(post.astype(np.int64), (0, pad))
        self.U_sign = np.pad(sign.astype(np.float64), (0, pad))
        self.U_origin = np.pad(np.zeros(self.size, dtype=np.int8), (0, pad))
        self.U_added_gen = np.pad(np.full(self.size, -1, dtype=np.int32), (0, pad))
        self.edge_index = {(int(p), int(q)): i for i, (p, q) in
                           enumerate(zip(self.U_pre[: self.size], self.U_post[: self.size]))}

        # 新边极性: 每神经元出边符号多数投票 (无出边默认兴奋)
        vot = np.bincount(self.U_pre[: self.size], weights=self.U_sign[: self.size],
                          minlength=n_nodes)
        self.nt_sign_node = np.where(vot >= 0, 1.0, -1.0)

        # ---- 种群: 初始真实边全部存在 ----
        self.mask = np.ones((pop_size, self._capacity), dtype=bool)
        self.weights = np.zeros((pop_size, self._capacity))
        self.weights[:, : self.size] = w_init[None, :] * self.rng.lognormal(
            0.0, 0.10, size=(pop_size, self.size)
        )

        self.add_log: list[tuple] = []   # (gen, pre, post, edge_id, reactivated)
        self.dup_log: list[tuple] = []   # (gen, src, dst, n_copied, n_old_cut)
        self.events_hist: list[dict] = []

    # ---- 宇宙维护 ----

    def _append_edge(self, pre: int, post: int, sign: float, origin: int, gen: int,
                     mask_row=None, w_row=None):
        if self.size >= self._capacity:
            grow = 1024
            for name in ("U_pre", "U_post", "U_sign", "U_origin", "U_added_gen"):
                setattr(self, name, np.pad(getattr(self, name), (0, grow)))
            self.mask = np.pad(self.mask, ((0, 0), (0, grow)))
            self.weights = np.pad(self.weights, ((0, 0), (0, grow)))
            # 正在构造的子代局部行也要同步扩容
            if mask_row is not None:
                mask_row = np.pad(mask_row, (0, grow))
            if w_row is not None:
                w_row = np.pad(w_row, (0, grow))
            self._capacity += grow
        idx = self.size
        self.U_pre[idx], self.U_post[idx] = pre, post
        self.U_sign[idx], self.U_origin[idx] = sign, origin
        self.U_added_gen[idx] = gen
        self.size += 1
        return idx, mask_row, w_row

    def payload(self, i: int) -> np.ndarray:
        """评估用有效权重 (不存在的边权重为 0), 与固定结构任务接口一致."""
        return self.weights[i, : self.size] * self.mask[i, : self.size]

    def edge_freq(self) -> np.ndarray:
        return self.mask[:, : self.size].mean(axis=0)

    def mean_effective(self) -> np.ndarray:
        """种群平均有效权重 (结构分析里的"阶段平均基因组")."""
        return (self.weights[:, : self.size] * self.mask[:, : self.size]).mean(axis=0)

    def diversity(self) -> float:
        ew = self.weights[:, : self.size] * self.mask[:, : self.size]
        return float(ew.std(axis=0).mean())

    def best_payload(self, fitness: np.ndarray) -> np.ndarray:
        i = int(np.argmax(fitness))
        return self.payload(i)

    # ---- 结构算子 ----

    def _op_add(self, gen: int, mask_row, w_row, events,
                offered: tuple | None = None):
        if offered is not None:
            pre, post = offered
        elif self.allowed_edges:
            pre, post = self.allowed_edges[int(self.rng.integers(len(self.allowed_edges)))]
        elif self.allowed_flows:
            flows = self.allowed_flows
            gpre, gpost = flows[int(self.rng.integers(len(flows)))]
            pre_pool = self._group_nodes.get(int(gpre))
            post_pool = self._group_nodes.get(int(gpost))
            if pre_pool is None or len(pre_pool) == 0 or post_pool is None or len(post_pool) == 0:
                return mask_row, w_row
            pre = int(self.rng.choice(pre_pool))
            post = int(self.rng.choice(post_pool))
        else:
            pre = int(self.rng.integers(self.n_nodes))
            post = int(self.rng.integers(self.n_nodes))
        if pre == post:
            return mask_row, w_row
        key = (pre, post)
        idx = self.edge_index.get(key)
        if idx is None:
            idx, mask_row, w_row = self._append_edge(
                pre, post, self.nt_sign_node[pre], self.ORIGIN_ADDED, gen,
                mask_row, w_row,
            )
            self.edge_index[key] = idx
        reactivated = bool(mask_row[idx])
        if not reactivated:
            mask_row[idx] = True
            w_row[idx] = max(self.new_edge_w, self.w_min)
            events["add"] += 1
        self.add_log.append((gen, pre, post, idx, reactivated))
        return mask_row, w_row

    def _op_del(self, mask_row, events) -> None:
        present = np.flatnonzero(mask_row[: self.size])
        if len(present) <= 100:  # 防止剪光线路
            return
        mask_row[int(self.rng.choice(present))] = False
        events["del"] += 1

    def _op_dup(self, gen: int, mask_row, w_row, events,
                offered: tuple | None = None):
        if offered is not None:
            s, t = offered
            if t < 0:  # 生成表时该源神经元无同群可复制目标: 与旧早退语义一致
                return mask_row, w_row
        else:
            s = int(self.rng.integers(self.n_nodes))
            grp = np.flatnonzero(self.group == self.group[s])
            grp = grp[grp != s]
            if len(grp) == 0:
                return mask_row, w_row
            t = int(self.rng.choice(grp))
        in_uni = slice(0, self.size)
        s_out = np.flatnonzero(mask_row[in_uni] & (self.U_pre[in_uni] == s))
        t_old = np.flatnonzero(mask_row[in_uni] & (self.U_pre[in_uni] == t))
        mask_row[t_old] = False  # 重复语义: t 的旧出边模式被 s 的覆写
        n_copy = 0
        for e in s_out:
            x = int(self.U_post[e])
            # 候选边模式下副本边也必须落在池内, 否则 dup 算子绕开搜索空间缩减
            if self._allowed_edge_set is not None and (t, x) not in self._allowed_edge_set:
                continue
            key = (t, x)
            idx = self.edge_index.get(key)
            if idx is None:
                # 副本继承源边的递质极性 (整个表型被复制)
                idx, mask_row, w_row = self._append_edge(
                    t, x, self.U_sign[e], self.ORIGIN_ADDED, gen, mask_row, w_row
                )
                self.edge_index[key] = idx
            if not mask_row[idx]:
                mask_row[idx] = True
            w_row[idx] = w_row[e]
            n_copy += 1
        events["dup"] += 1
        self.dup_log.append((gen, s, t, n_copy, len(t_old)))
        return mask_row, w_row

    # ---- 代际更迭 ----

    def _pick(self, fitness: np.ndarray) -> int:
        cand = self.rng.integers(0, self.pop_size, size=self.tournament)
        return int(cand[np.argmax(fitness[cand])])

    def evolve(self, fitness: np.ndarray, gen: int) -> dict:
        rng = self.rng
        # 精英原样保留
        elite_rows = np.argsort(fitness)[::-1][: self.elite]
        self.mask[: self.elite] = self.mask[elite_rows]
        self.weights[: self.elite] = self.weights[elite_rows]

        events = {"add": 0, "del": 0, "dup": 0}

        # ---- P-A 信用分配 (R7 §10) ----
        # 反事实信用: credit[e] = 携带 e 的个体平均适应度 − 不携带 e 的平均适应度.
        # 只用本代观测量 (不引入"哪条边应该好"的先验知识), 相当于对每条新增边做
        # 一次群体内 ablation. 关闭时 credit_norm=None, 交叉路径与历史完全一致.
        credit_norm = None
        if self.credit_gain > 0.0 and self.size > self.n_base:
            M = self.mask[:, self.n_base : self.size].astype(np.float64)
            f = np.asarray(fitness, dtype=np.float64)[:, None]
            n_car = M.sum(axis=0)
            n_non = float(self.pop_size) - n_car
            with np.errstate(invalid="ignore", divide="ignore"):
                mu_car = np.where(n_car > 0, (M * f).sum(axis=0) / np.maximum(n_car, 1.0), 0.0)
                mu_non = np.where(
                    n_non > 0, ((1.0 - M) * f).sum(axis=0) / np.maximum(n_non, 1.0), 0.0
                )
            credit = mu_car - mu_non
            # 只在"两侧都有样本"的边上取信用: 单侧为空的边 (n_car=0 即该边已从群体
            # 消失, 或 n_non=0 即已全员携带) 信息不完整, 置 0 且**不参与尺度估计** ——
            # 否则这些边会被算成 ±全体均值 (~数十), 把 sd 抬高数倍, 从而把真实信用
            # 信号压缩掉, 使信用分配形同虚设.
            valid = (n_car > 0) & (n_non > 0)
            credit = np.where(valid, credit, 0.0)
            # 单代 credit 的样本量极小 (实测每条新边只有 0~2 个携带者), 其标准误与
            # 真实效应同量级. 跨代指数累积可按 sqrt(有效代数) 降噪 —— 即资格迹
            # (eligibility trace): 对"持续表现好"的边累积正证据, 对偶发噪声不敏感.
            if self.credit_mode == "reward_ema":
                n_new = int(self.size) - self.n_base
                if self._credit_ema is None or len(self._credit_ema) < n_new:
                    old = self._credit_ema if self._credit_ema is not None else np.zeros(0)
                    self._credit_ema = np.concatenate([old, np.zeros(n_new - len(old))])
                seg = self._credit_ema[:n_new]
                seg[:] = self.credit_ema_decay * seg + (1.0 - self.credit_ema_decay) * credit
                src = seg
            else:
                src = credit
            sd = float(src[valid].std()) if bool(valid.any()) else 0.0
            if sd > 0:
                credit_norm = np.clip(src / sd, -4.0, 4.0)

        for child in range(self.elite, self.pop_size):
            pa, pb = self._pick(fitness), self._pick(fitness)
            cm = rng.random(self.size) < 0.5
            m = np.where(cm, self.mask[pa, : self.size], self.mask[pb, : self.size])
            # P-A: 新增边的继承概率按反事实信用加权. 均值 ≈0 → 总保留率不变,
            # 只是"有偏": 有益边更易留下, 无用边更易丢失. 原始边不受影响.
            # 注意: self.size 会在本子代循环内因 _op_add 而增长, 故 credit_norm 只
            # 覆盖"本代算信用时已存在的边"; 循环中新加入的边无信用, p_keep=1 (不干预).
            if credit_norm is not None:
                n_c = int(len(credit_norm))
                p_keep = np.ones(self.size)
                raw = self.credit_gain * credit_norm
                if self.credit_mode in ("reward_only", "reward_ema"):
                    # 只奖励、不惩罚: 保证 p_keep >= 0.5, 新边的平均保留率不被压低.
                    # 初版 symmetric 的惩罚项压过了信号 —— 新边在加入初期普遍拖累
                    # 适应度 (credit<0), 被系统性淘汰, 而有益边也需要时间才显形.
                    raw = np.maximum(raw, 0.0)
                p_keep[self.n_base : self.n_base + n_c] = np.clip(0.5 + raw, 0.05, 0.95)
                m = m & (rng.random(self.size) < p_keep)
            if self.weight_crossover:
                w = np.where(cm, self.weights[pa, : self.size], self.weights[pb, : self.size])
            else:
                w = self.weights[pa, : self.size].copy()
            w = np.clip(w * rng.normal(1.0, self.mut_sigma, size=self.size),
                        self.w_min, self.w_max)

            mask_row = np.zeros(self._capacity, dtype=bool)
            mask_row[: self.size] = m
            w_row = np.zeros(self._capacity)
            w_row[: self.size] = w

            # spike-in: 该 (gen, child) 的供给侧决策来自共享提议表时, 开火判定/
            # 加边提议/重复源目标都不再消耗 self.rng; 剪边目标与选择仍各臂自理。
            # gen 沿用 evolve 的 1 基全局代数 (与 add_log 同口径); 缺行 = 表与
            # 协议不一致, 响亮失败, 不许静默回退 self.rng 供给。
            if self.offer_table:
                try:
                    offer = self.offer_table[(gen, child)]
                except KeyError:
                    raise RuntimeError(
                        f"提议表缺 (gen={gen}, child={child}) 行 —— 表与协议不一致, 拒绝静默回退"
                    )
            else:
                offer = None
            if offer is not None:
                if offer["do_del"]:
                    self._op_del(mask_row, events)
                if offer["do_add"]:
                    mask_row, w_row = self._op_add(
                        gen, mask_row, w_row, events,
                        offered=(int(offer["pre"]), int(offer["post"])),
                    )
                if offer["do_dup"]:
                    mask_row, w_row = self._op_dup(
                        gen, mask_row, w_row, events,
                        offered=(int(offer["dup_src"]), int(offer["dup_dst"])),
                    )
            else:
                if rng.random() < self.p_del:
                    self._op_del(mask_row, events)
                if rng.random() < self.p_add:
                    mask_row, w_row = self._op_add(gen, mask_row, w_row, events)
                if rng.random() < self.p_dup:
                    mask_row, w_row = self._op_dup(gen, mask_row, w_row, events)

            # P-B 捆绑上限: 只作用新增段 [n_base, size); 超额随机保留 cap 条。
            if self.bundle_cap > 0:
                new_idx = np.flatnonzero(mask_row[self.n_base: self.size]) + self.n_base
                if len(new_idx) > self.bundle_cap:
                    keep = self.rng.choice(new_idx, size=self.bundle_cap, replace=False)
                    mask_row[np.setdiff1d(new_idx, keep)] = False

            self.mask[child, : len(mask_row)] = mask_row
            self.weights[child, : len(w_row)] = w_row
        self.events_hist.append(events)
        # 注意: 刻意不做 GC。死列(被所有个体丢弃的新增边)对评估贡献为 0,
        # 而 GC 的列压缩会重编号边 id, 破坏跨代频率追踪。40 代先导的死列
        # 累积约几百列, 内存与计算代价可忽略。
        return events
