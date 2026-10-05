"""身体-大脑协同进化模块.

在已有 Fly-Evo 框架基础上扩展:
- MorphologyConfig: 可进化的身体形态参数 (体型/触角/复眼/翅/代谢/脑重比)
- EmbodiedGA: 组合基因组进化算法 (突触权重 + 形态参数联合进化)
- 扩展代谢模型: 体型异速生长 (Kleiber^0.75) + 脑能耗 + 运动能耗
- 形态感知的评估函数: 根据个体形态动态构建 WorldConfig 和 SensorimotorMapping

设计原则:
- 与已有 GeneticAlgorithm / LIFBrain / World 接口完全兼容
- 形态参数有生物学合理的范围约束
- 能量预算是硬约束, 防止"无限大脑"的不合理进化
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .brain import BrainConfig, LIFBrain
from .agent import SensorimotorMapping, evaluate_episode
from . import world as W


# ============================================================================
# 1. 形态参数定义
# ============================================================================

# 形态参数的名称、默认值、最小值、最大值
# 这些范围基于真实果蝇的测量值和合理的进化变异空间
MORPH_PARAMS = [
    # (name, default, min, max, description)
    ("body_size",       1.0,  0.5,  3.0,  "体型缩放: 影响代谢/能量容量/运动惯性"),
    ("antenna_dist",    3.0,  1.0,  8.0,  "触角距离: 影响嗅觉空间分辨率和转向灵敏度"),
    ("antenna_angle",   0.9,  0.3,  1.4,  "触角偏角(rad): 影响左右气味对比"),
    ("eye_size",        1.0,  0.5,  3.0,  "复眼大小: 影响视觉灵敏度(简化为感觉增益调制)"),
    ("wing_size",       1.0,  0.3,  3.0,  "翅大小: 影响飞行能力(>1.5可飞行)和能耗"),
    ("metab_rate",      4.0,  1.0, 12.0,  "基础代谢率(能量/s): 高代谢=快饿死但可能更活跃"),
    ("brain_energy",    0.02, 0.005,0.15, "脑能耗占比: 大脑活动的能量消耗系数"),
    ("sens_gain",     600.0, 200.0,2000.0,"感觉增益: ORN电流注入强度"),
    ("speed_gain",      0.06, 0.02, 0.20, "运动增益: DN发放到轮速的转换"),
    ("speed_max",       8.0,  3.0, 20.0,  "最大速度: 运动上限"),
    ("turn_gain",       0.04, 0.01, 0.15, "转向增益: 左右DN差异到转向的转换"),
]

MORPH_NAMES = [p[0] for p in MORPH_PARAMS]
MORPH_DEFAULTS = np.array([p[1] for p in MORPH_PARAMS])
MORPH_MINS = np.array([p[2] for p in MORPH_PARAMS])
MORPH_MAXS = np.array([p[3] for p in MORPH_PARAMS])
N_MORPH = len(MORPH_PARAMS)


@dataclass
class Morphology:
    """果蝇身体形态参数.

    所有参数都是可进化的连续值, 有生物学合理的范围约束.
    用向量表示便于进化算法操作.
    """
    body_size: float = 1.0
    antenna_dist: float = 3.0
    antenna_angle: float = 0.9
    eye_size: float = 1.0
    wing_size: float = 1.0
    metab_rate: float = 4.0
    brain_energy: float = 0.02
    sens_gain: float = 600.0
    speed_gain: float = 0.06
    speed_max: float = 8.0
    turn_gain: float = 0.04

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "Morphology":
        v = np.clip(v, MORPH_MINS, MORPH_MAXS)
        return cls(**{name: float(val) for name, val in zip(MORPH_NAMES, v)})

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in MORPH_NAMES])

    def can_fly(self) -> bool:
        """翅大小超过阈值才能飞行."""
        return self.wing_size >= 1.5

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# 2. 形态 → 世界配置 / 感觉运动映射 的转换
# ============================================================================

def morphology_to_world_config(base_cfg: W.WorldConfig, morph: Morphology) -> W.WorldConfig:
    """根据形态参数动态构建世界配置.

    关键生物学约束:
    - 体型越大, 能量容量越大 (异速生长: 能量容量 ∝ body_size^1.0)
    - 体型越大, 基础代谢越高 (Kleiber定律: 代谢 ∝ body_size^0.75)
    - 触角距离影响嗅觉空间分辨率
    - 翅大小影响运动能力 (大翅=更高速但更耗能)
    """
    cfg = W.WorldConfig(**{k: v for k, v in base_cfg.__dict__.items()})

    # 能量容量随体型线性增长 (更大的身体=更多脂肪储备)
    cfg.energy_cap = base_cfg.energy_cap * morph.body_size
    cfg.energy0 = base_cfg.energy0 * morph.body_size

    # 基础代谢: Kleiber定律 (代谢率 ∝ 体重^0.75) + 个体代谢率参数
    cfg.metab_base = morph.metab_rate * (morph.body_size ** 0.75)

    # 运动代谢系数: 大翅飞行更耗能, 大体型运动更耗能
    flight_penalty = 1.0 + 0.5 * max(0, morph.wing_size - 1.0)  # 翅越大运动能耗越高
    cfg.metab_speed = base_cfg.metab_speed * morph.body_size * flight_penalty

    # 触角参数
    cfg.antenna_dist = morph.antenna_dist
    cfg.antenna_angle = morph.antenna_angle

    # 食物能量: 大体型需要更多食物, 但食物能量值不变 (选择压力来自代谢)
    return cfg


def morphology_to_mapping(base_mapping: SensorimotorMapping, morph: Morphology) -> SensorimotorMapping:
    """根据形态参数构建感觉运动映射.

    - 复眼大小和感觉增益影响感觉输入强度
    - 运动增益和最大速度影响运动输出
    - 转向增益影响转向灵敏度
    """
    m = SensorimotorMapping(
        n_nodes=base_mapping.n_nodes,
        food_l=base_mapping.food_l,
        food_r=base_mapping.food_r,
        toxin_l=base_mapping.toxin_l,
        toxin_r=base_mapping.toxin_r,
        pred_l=base_mapping.pred_l,
        pred_r=base_mapping.pred_r,
        dn_left=base_mapping.dn_left,
        dn_right=base_mapping.dn_right,
        dn_center=base_mapping.dn_center,
        # 形态参数直接映射
        sens_gain=morph.sens_gain * (0.5 + 0.5 * morph.eye_size),  # 复眼越大感觉越灵敏
        speed_gain=morph.speed_gain,
        speed_base=3.0 * (0.5 + 0.5 * morph.wing_size),  # 大翅基础速度更高
        speed_max=morph.speed_max,
        turn_gain=morph.turn_gain,
    )
    return m


# ============================================================================
# 3. 扩展代谢: 脑能耗计算
# ============================================================================

def compute_brain_energy_cost(
    spike_counts: np.ndarray,
    morph: Morphology,
    n_steps: int,
    dt: float,
) -> float:
    """计算大脑在一个回合中的能量消耗.

    模型: 脑能耗 = brain_energy系数 × 总发放数 × 体型缩放
    这是一个简化模型,  captures the key idea that more neural activity = more energy.
    真实果蝇脑能耗约占总代谢的10-20%, 这里用brain_energy参数控制.
    """
    total_spikes = float(spike_counts.sum())
    # 脑能耗校准: 真实果蝇脑能耗占基础代谢10-20%
    # 基础代谢~4能量/s, 10s回合~40能量, 脑能耗应~4-8
    # 4409神经元*平均~50Hz*10s≈220万发放, 故缩放因子取0.00001
    cost_per_spike = morph.brain_energy * morph.body_size
    return total_spikes * cost_per_spike * 0.00001


# ============================================================================
# 4. 组合基因组进化算法
# ============================================================================

class EmbodiedGA:
    """身体-大脑组合基因组的进化算法.

    每个个体的基因组 = (突触权重向量, 形态参数向量)
    两部分独立进行选择/交叉/突变, 但共享同一个适应度评估.

    权重部分: 与已有 GeneticAlgorithm 一致 (均匀交叉 + 高斯突变)
    形态部分: 每个参数独立交叉 (BLX-alpha) + 高斯突变
    """

    def __init__(
        self,
        w_init: np.ndarray,
        pop_size: int = 60,
        elite: int = 2,
        tournament: int = 3,
        w_mut_sigma: float = 0.15,
        morph_mut_sigma: float = 0.08,  # 形态突变用相对标准差
        w_min: float = 0.01,
        w_max: float = 15.0,
        seed: int = 0,
        evolve_body: bool = True,  # 是否允许身体进化 (False=固定野生型身体)
        evolve_brain: bool = True,  # 是否允许大脑进化 (False=固定野生型权重)
    ):
        self.w_init = w_init
        self.pop_size = pop_size
        self.elite = elite
        self.tournament = tournament
        self.w_mut_sigma = w_mut_sigma
        self.morph_mut_sigma = morph_mut_sigma
        self.w_min = w_min
        self.w_max = w_max
        self.evolve_body = evolve_body
        self.evolve_brain = evolve_brain
        self.rng = np.random.default_rng(seed)

        n_w = len(w_init)

        # 权重基因组: 初始 = 真实权重 * 对数正态扰动
        self.w_genomes = w_init[None, :] * self.rng.lognormal(
            0.0, 0.10, size=(pop_size, n_w)
        )
        self.w_genomes = np.clip(self.w_genomes, w_min, w_max)

        # 形态基因组: 初始 = 默认值 * 小扰动
        self.morph_genomes = MORPH_DEFAULTS[None, :] * self.rng.lognormal(
            0.0, 0.05, size=(pop_size, N_MORPH)
        )
        self.morph_genomes = np.clip(self.morph_genomes, MORPH_MINS, MORPH_MAXS)

    def diversity(self) -> dict:
        """种群多样性: 权重和形态分别计算."""
        w_div = float((self.w_genomes.std(axis=0) / self.w_init).mean())
        m_div = float((self.morph_genomes.std(axis=0) / MORPH_DEFAULTS).mean())
        return {"weight_div": w_div, "morph_div": m_div}

    def get_morphology(self, i: int) -> Morphology:
        return Morphology.from_vector(self.morph_genomes[i])

    def get_weights(self, i: int) -> np.ndarray:
        return self.w_genomes[i]

    def evolve(self, fitness: np.ndarray) -> dict:
        """选择/交叉/变异, 生成下一代. 返回事件统计."""
        rng = self.rng
        order = np.argsort(fitness)[::-1]

        # 精英保留
        elite_w = self.w_genomes[order[:self.elite]].copy()
        elite_m = self.morph_genomes[order[:self.elite]].copy()

        def pick() -> int:
            cand = rng.integers(0, self.pop_size, size=self.tournament)
            return int(cand[np.argmax(fitness[cand])])

        children_w = []
        children_m = []
        events = {"w_mut": 0, "m_mut": 0, "cross": 0}

        while len(children_w) < self.pop_size - self.elite:
            pa, pb = pick(), pick()

            # --- 权重交叉: 均匀交叉 ---
            if self.evolve_brain:
                mask = rng.random(len(self.w_init)) < 0.5
                child_w = np.where(mask, self.w_genomes[pa], self.w_genomes[pb])
                child_w = child_w * rng.normal(1.0, self.w_mut_sigma, size=child_w.shape)
                child_w = np.clip(child_w, self.w_min, self.w_max)
                events["w_mut"] += 1
            else:
                child_w = self.w_init.copy()  # 固定野生型权重

            # --- 形态交叉: BLX-alpha (每个参数独立) ---
            if self.evolve_body:
                alpha = 0.5
                cmin = np.minimum(self.morph_genomes[pa], self.morph_genomes[pb])
                cmax = np.maximum(self.morph_genomes[pa], self.morph_genomes[pb])
                interval = cmax - cmin
                child_m = rng.uniform(
                    cmin - alpha * interval,
                    cmax + alpha * interval,
                )
                # 形态突变: 相对高斯突变
                child_m = child_m * rng.normal(1.0, self.morph_mut_sigma, size=N_MORPH)
                child_m = np.clip(child_m, MORPH_MINS, MORPH_MAXS)
                events["m_mut"] += 1
            else:
                child_m = MORPH_DEFAULTS.copy()  # 固定野生型形态

            events["cross"] += 1
            children_w.append(child_w)
            children_m.append(child_m)

        self.w_genomes = np.vstack([elite_w, np.array(children_w)])
        self.morph_genomes = np.vstack([elite_m, np.array(children_m)])
        return events

    def best(self, fitness: np.ndarray) -> tuple[int, np.ndarray, Morphology]:
        i = int(np.argmax(fitness))
        return i, self.w_genomes[i], Morphology.from_vector(self.morph_genomes[i])

    def mean_morphology(self) -> Morphology:
        return Morphology.from_vector(self.morph_genomes.mean(axis=0))


# ============================================================================
# 5. 形态感知的单回合评估
# ============================================================================

def evaluate_embodied_episode(
    brain: LIFBrain,
    base_mapping: SensorimotorMapping,
    base_world_cfg: W.WorldConfig,
    weights: np.ndarray,
    morph: Morphology,
    episode_seed: int,
    noise_seed: int,
    record_every: int = 0,
    track_activity: bool = False,
    include_brain_energy: bool = True,
) -> tuple[dict, list, Optional[np.ndarray]]:
    """评估一个具身个体的一回合.

    与 evaluate_episode 的区别:
    1. 根据形态参数动态构建 WorldConfig 和 SensorimotorMapping
    2. 可选地计算脑能耗并从能量中扣除
    3. 返回形态相关的统计信息
    """
    # 应用形态到世界和映射
    world_cfg = morphology_to_world_config(base_world_cfg, morph)
    mapping = morphology_to_mapping(base_mapping, morph)

    # 设置脑权重
    brain.set_weights(weights)

    # 运行标准回合
    stats, frames, activity = evaluate_episode(
        brain, mapping, world_cfg, episode_seed, noise_seed,
        record_every=record_every, track_activity=track_activity,
    )

    # 脑能耗扣除 (如果需要)
    if include_brain_energy and activity is not None:
        # activity 是平均发放率(Hz), 估算总发放数
        n_steps = int(world_cfg.duration / world_cfg.dt_ctrl)
        steps_per_tick = max(int(round(world_cfg.dt_ctrl / brain.dt)), 1)
        total_spikes_est = activity * n_steps * steps_per_tick * brain.dt
        brain_cost = compute_brain_energy_cost(
            total_spikes_est, morph, n_steps * steps_per_tick, brain.dt
        )
        stats["brain_energy_cost"] = round(brain_cost, 2)
        stats["final_energy_after_brain"] = round(stats["final_energy"] - brain_cost, 1)
    else:
        stats["brain_energy_cost"] = 0.0
        stats["final_energy_after_brain"] = stats["final_energy"]

    # 添加形态信息到统计
    stats["body_size"] = round(morph.body_size, 3)
    stats["wing_size"] = round(morph.wing_size, 3)
    stats["brain_energy_ratio"] = round(morph.brain_energy, 4)
    stats["can_fly"] = int(morph.can_fly())

    return stats, frames, activity
