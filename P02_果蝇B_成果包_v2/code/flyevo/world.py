"""2D 拟真环境: 气味场(食物/毒物/掠食者) + 实体 + 能量代谢.

世界以控制窗粒度(dt_ctrl=10ms)更新; 感觉以高斯气味场呈现, 神经元
只通过 ORN 感受气味, 毒物/掠食者接触则直接作用于身体(能量/死亡).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TWO_PI = 2 * np.pi


@dataclass
class WorldConfig:
    width: float = 100.0
    height: float = 100.0
    duration: float = 20.0        # 回合时长 (s)
    dt_ctrl: float = 0.01         # 控制步长 (s)
    antenna_dist: float = 3.0     # 触角到身体中心距离
    antenna_angle: float = 0.9    # 触角相对朝向的偏角 (rad, 约 52°)

    # 代谢
    energy0: float = 100.0
    energy_cap: float = 160.0
    metab_base: float = 4.0       # 基础代谢 (能量/s); 100 能量约 20s, 不觅食会饿死
    metab_speed: float = 0.25     # 运动代谢系数 (能量/s / 速度)
    # 结构突变代价 (默认 0 = 关闭, 不影响既有行为):
    # 每条"超出基线"的突触持续消耗能量, 使无用加边净收益为负 -> 让"固定"重新具有
    # 选择性含义. 依据 (2026-09-25 诊断): 原系统加边零代价 + 删边概率低, 导致边频率
    # 94% 单调上升 (中性棘轮), 使"固定率"无法区分有益边与诱饵边 (47.3% vs 47.2%).
    metab_per_extra_edge: float = 0.0
    n_base_edges: int = 0         # 基线突触数; 配合上项使用 (0 = 全部突触计入成本)

    # 食物
    n_food: int = 30
    food_energy: float = 25.0
    food_radius: float = 2.5
    food_odor_sigma: float = 14.0
    food_regrow_interval: float = 1.0  # 每秒补 1 颗, 补到上限

    # 毒物
    n_toxin: int = 8
    toxin_radius: float = 4.0
    toxin_damage: float = 25.0    # 能量/s (接触时)
    toxin_odor_sigma: float = 7.0

    # 掠食者 (stage>=3 才出现)
    predator_speed: float = 3.0
    predator_radius: float = 3.0
    predator_odor_sigma: float = 6.0

    # 感觉
    sensor_noise: float = 0.03
    wander_std: float = 0.8   # 行走朝向抖动 (rad/sqrt(s))

    rng_seed: int = 0


@dataclass
class AgentState:
    pos: np.ndarray            # [x, y]
    heading: float
    speed: float
    energy: float
    alive: bool
    time_alive: float


@dataclass
class WorldState:
    food_pos: np.ndarray       # (n_food, 2)
    toxin_pos: np.ndarray
    predator_pos: np.ndarray | None = None
    predator_active: bool = False
    agent: AgentState | None = None
    food_eaten: int = 0
    toxin_exposure_s: float = 0.0
    predator_near_min: float = 1e9   # 掠食者最近距离
    t: float = 0.0
    n_extra_edges: int = 0           # 本个体超出基线的突触数 (结构代价用, 见 WorldConfig)


def reset_world(cfg: WorldConfig, rng: np.random.Generator, with_predator: bool) -> WorldState:
    s = WorldState(
        food_pos=rng.uniform(4, cfg.width - 4, size=(cfg.n_food, 2)),
        toxin_pos=rng.uniform(4, cfg.width - 4, size=(cfg.n_toxin, 2)),
        predator_pos=rng.uniform(4, cfg.width - 4, size=2) if with_predator else None,
        predator_active=with_predator,
        agent=AgentState(
            pos=np.array([cfg.width / 2, cfg.height / 2]),
            heading=rng.uniform(0, TWO_PI),
            speed=0.0,
            energy=cfg.energy0,
            alive=True,
            time_alive=0.0,
        ),
    )
    return s


def odor_concentration(
    pos: np.ndarray, sources: np.ndarray, sigma: float
) -> float:
    """多个高斯源在 pos 处的浓度和, 截断到 [0,1]."""
    if len(sources) == 0:
        return 0.0
    d2 = ((sources - pos) ** 2).sum(axis=1)
    return float(min(np.exp(-d2 / (2 * sigma**2)).sum(), 1.0))


def sensor_readings(cfg: WorldConfig, s: WorldState) -> dict:
    """双侧触角处采样三种气味; 返回 {food,toxin,predator} x {L,R} 浓度."""
    a = s.agent
    offs = np.array(
        [
            [np.cos(a.heading + cfg.antenna_angle), np.sin(a.heading + cfg.antenna_angle)],
            [np.cos(a.heading - cfg.antenna_angle), np.sin(a.heading - cfg.antenna_angle)],
        ]
    ) * cfg.antenna_dist
    ant_l = a.pos + offs[0]
    ant_r = a.pos + offs[1]
    pred_src = s.predator_pos[None, :] if s.predator_active else np.zeros((0, 2))
    out = {}
    for name, src, sig in [
        ("food", s.food_pos, cfg.food_odor_sigma),
        ("toxin", s.toxin_pos, cfg.toxin_odor_sigma),
        ("predator", pred_src, cfg.predator_odor_sigma),
    ]:
        out[name] = (
            odor_concentration(ant_l, src, sig),
            odor_concentration(ant_r, src, sig),
        )
    return out


def move_agent(cfg: WorldConfig, s: WorldState, v_left: float, v_right: float, rng: np.random.Generator) -> None:
    """差速驱动: vL/vR 为左右轮速, 朝向由轮速差 + 行走抖动决定.

    朝向随机游走是果蝇觅食的真实特征 (随机游走 + 气味引导转向), 也保证
    未进化个体的行为有个体差异, 让选择有东西可挑.
    """
    a = s.agent
    a.speed = max((v_left + v_right) / 2.0, 0.0)
    a.heading += (v_right - v_left) / (2.0 * cfg.antenna_dist) * cfg.dt_ctrl
    a.heading += rng.normal(0.0, cfg.wander_std) * np.sqrt(cfg.dt_ctrl)
    a.heading = (a.heading + TWO_PI) % TWO_PI
    a.pos = a.pos + np.array([np.cos(a.heading), np.sin(a.heading)]) * a.speed * cfg.dt_ctrl
    a.pos = np.clip(a.pos, 1.0, [cfg.width - 1.0, cfg.height - 1.0])


def apply_contacts(cfg: WorldConfig, s: WorldState, rng: np.random.Generator) -> None:
    """吃食物 / 中毒 / 被捕食, 并结算代谢."""
    a = s.agent
    dtc = cfg.dt_ctrl

    # 吃: 与食物圆相交
    d = np.linalg.norm(s.food_pos - a.pos, axis=1)
    hit = d < cfg.food_radius + 1.0
    if hit.any():
        a.energy = min(a.energy + cfg.food_energy * hit.sum(), cfg.energy_cap)
        s.food_eaten += int(hit.sum())
        # 被吃掉的食物直接挪到随机新位置 (相当于环境再生)
        s.food_pos[hit] = rng.uniform(4, cfg.width - 4, size=(int(hit.sum()), 2))

    # 毒: 接触期内持续掉能量
    d = np.linalg.norm(s.toxin_pos - a.pos, axis=1)
    in_toxin = (d < cfg.toxin_radius + 1.0).any()
    if in_toxin:
        a.energy -= cfg.toxin_damage * dtc
        s.toxin_exposure_s += dtc

    # 掠食者: 追踪 + 接触即死
    if s.predator_active:
        s.predator_near_min = min(
            s.predator_near_min, float(np.linalg.norm(s.predator_pos - a.pos))
        )
        to_agent = a.pos - s.predator_pos
        dist = np.linalg.norm(to_agent)
        if dist < cfg.predator_radius + 1.0:
            a.alive = False
        else:
            step = cfg.predator_speed * dtc
            s.predator_pos = s.predator_pos + to_agent / dist * step
            s.predator_pos += rng.normal(0, step * 0.3, size=2)
            s.predator_pos = np.clip(s.predator_pos, 1.0, [cfg.width - 1, cfg.height - 1])

    # 代谢
    # 基础代谢 + 运动代谢 + 结构维持成本 (结构项默认 0, 不改变既有行为)
    metab = cfg.metab_base + cfg.metab_speed * a.speed
    if cfg.metab_per_extra_edge > 0.0:
        metab += cfg.metab_per_extra_edge * s.n_extra_edges
    a.energy -= metab * dtc
    a.time_alive += dtc
    if a.energy <= 0:
        a.alive = False
