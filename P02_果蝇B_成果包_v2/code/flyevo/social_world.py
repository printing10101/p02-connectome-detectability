"""桥2: 多果蝇社会世界 — 信息素信号 + 有偿带宽.

对齐 GlossoGen: 信息不对称 + 字符计费 → 协议压缩.
果蝇版: 食物标记/警戒信息素; DN 子集发射, 释放量扣能量.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import world as W


@dataclass
class SocialWorldConfig(W.WorldConfig):
    n_fly: int = 2
    signal_cost: float = 0.15
    pheromone_decay: float = 0.4
    pheromone_sigma: float = 8.0
    pheromone_gain: float = 80.0
    group_lambda: float = 0.3
    max_release: float = 1.0
    signal_channels: int = 2


@dataclass
class FlyState:
    pos: np.ndarray
    heading: float
    speed: float
    energy: float
    alive: bool
    food_eaten: int = 0
    signal_budget_spent: float = 0.0
    release: float = 0.0
    channel: int = 0


@dataclass
class SocialWorldState:
    food_pos: np.ndarray
    toxin_pos: np.ndarray
    predator_pos: np.ndarray | None
    predator_active: bool
    flies: list[FlyState]
    pheromones: list = field(default_factory=list)
    t: float = 0.0
    food_eaten_total: int = 0


def reset_social(cfg: SocialWorldConfig, rng: np.random.Generator, with_predator: bool) -> SocialWorldState:
    food = rng.uniform(4, cfg.width - 4, size=(cfg.n_food, 2))
    toxin = rng.uniform(4, cfg.width - 4, size=(cfg.n_toxin, 2))
    pred = rng.uniform(4, cfg.width - 4, size=2) if with_predator else None
    flies = []
    for _ in range(cfg.n_fly):
        flies.append(
            FlyState(
                pos=np.array([cfg.width / 2, cfg.height / 2]) + rng.normal(0, 3, 2),
                heading=rng.uniform(0, 2 * np.pi),
                speed=0.0,
                energy=cfg.energy0,
                alive=True,
            )
        )
    return SocialWorldState(
        food_pos=food,
        toxin_pos=toxin,
        predator_pos=pred,
        predator_active=with_predator,
        flies=flies,
        pheromones=[],
    )


def pheromone_concentration(pos: np.ndarray, particles: list, channel: int, sigma: float) -> float:
    if not particles:
        return 0.0
    arr = np.asarray([p for p in particles if p[3] == channel], dtype=float)
    if len(arr) == 0:
        return 0.0
    d2 = ((arr[:, :2] - pos) ** 2).sum(axis=1)
    w = arr[:, 2] * np.exp(-d2 / (2 * sigma**2))
    return float(min(w.sum(), 1.0))


def social_sensor_readings(cfg: SocialWorldConfig, s: SocialWorldState, fly_i: int) -> dict:
    a = s.flies[fly_i]
    offs = np.array(
        [
            [np.cos(a.heading + cfg.antenna_angle), np.sin(a.heading + cfg.antenna_angle)],
            [np.cos(a.heading - cfg.antenna_angle), np.sin(a.heading - cfg.antenna_angle)],
        ]
    ) * cfg.antenna_dist
    ant_l, ant_r = a.pos + offs[0], a.pos + offs[1]
    pred_src = s.predator_pos[None, :] if s.predator_active else np.zeros((0, 2))
    out = {}
    for name, src, sig in [
        ("food", s.food_pos, cfg.food_odor_sigma),
        ("toxin", s.toxin_pos, cfg.toxin_odor_sigma),
        ("predator", pred_src, cfg.predator_odor_sigma),
    ]:
        out[name] = (
            W.odor_concentration(ant_l, src, sig),
            W.odor_concentration(ant_r, src, sig),
        )
    for ch, key in enumerate(["sig_food", "sig_alarm"]):
        out[key] = (
            pheromone_concentration(ant_l, s.pheromones, ch, cfg.pheromone_sigma),
            pheromone_concentration(ant_r, s.pheromones, ch, cfg.pheromone_sigma),
        )
    return out


def step_social(
    cfg: SocialWorldConfig,
    s: SocialWorldState,
    wheels: list,
    releases: list,
    rng: np.random.Generator,
) -> SocialWorldState:
    dt = cfg.dt_ctrl
    if s.pheromones:
        new_p = []
        for p in s.pheromones:
            amt = p[2] * (1.0 - cfg.pheromone_decay * dt)
            if amt > 1e-4:
                new_p.append([p[0], p[1], amt, p[3]])
        s.pheromones = new_p

    if s.predator_active and s.predator_pos is not None:
        alive = [f for f in s.flies if f.alive]
        if alive:
            tgt = min(alive, key=lambda f: np.linalg.norm(f.pos - s.predator_pos))
            d = tgt.pos - s.predator_pos
            n = np.linalg.norm(d)
            if n > 1e-6:
                s.predator_pos = s.predator_pos + (d / n) * cfg.predator_speed * dt

    for i, fly in enumerate(s.flies):
        if not fly.alive:
            continue
        vl, vr = wheels[i]
        speed = (vl + vr) / 2.0
        ang = (vr - vl) / max(cfg.antenna_dist, 1e-6) * 0.5
        fly.speed = float(np.clip(speed, 0, 8.0))
        fly.heading = float(fly.heading + ang * dt + rng.normal(0, cfg.wander_std * np.sqrt(dt)))
        fly.pos = fly.pos + np.array([np.cos(fly.heading), np.sin(fly.heading)]) * fly.speed * dt
        fly.pos[0] = float(np.clip(fly.pos[0], 0, cfg.width))
        fly.pos[1] = float(np.clip(fly.pos[1], 0, cfg.height))

        amt, ch = releases[i]
        amt = float(np.clip(amt, 0.0, cfg.max_release))
        if amt > 0:
            s.pheromones.append([float(fly.pos[0]), float(fly.pos[1]), amt, int(ch)])
            fly.signal_budget_spent += amt
            fly.energy -= cfg.signal_cost * amt
            fly.release = amt
            fly.channel = int(ch)
        else:
            fly.release = 0.0

        fly.energy -= (cfg.metab_base + cfg.metab_speed * fly.speed) * dt

        if len(s.food_pos):
            d = np.linalg.norm(s.food_pos - fly.pos, axis=1)
            hit = np.flatnonzero(d < cfg.food_radius)
            if len(hit):
                s.food_eaten_total += 1
                fly.food_eaten += 1
                fly.energy = min(cfg.energy_cap, fly.energy + cfg.food_energy)
                s.food_pos = np.delete(s.food_pos, hit[0], axis=0)

        if len(s.toxin_pos):
            d = np.linalg.norm(s.toxin_pos - fly.pos, axis=1)
            if np.any(d < cfg.toxin_radius):
                fly.energy -= cfg.toxin_damage * dt

        if s.predator_active and s.predator_pos is not None:
            if np.linalg.norm(fly.pos - s.predator_pos) < cfg.predator_radius:
                fly.alive = False

        if fly.energy <= 0:
            fly.alive = False

    s.t += dt
    return s


def social_stats(s: SocialWorldState) -> dict:
    foods = [f.food_eaten for f in s.flies]
    alive = [f.alive for f in s.flies]
    spent = [f.signal_budget_spent for f in s.flies]
    return {
        "food": int(sum(foods)),
        "food_mean": float(np.mean(foods)) if foods else 0.0,
        "alive_frac": float(np.mean(alive)) if alive else 0.0,
        "signal_spent_mean": float(np.mean(spent)) if spent else 0.0,
        "n_alive": int(sum(alive)),
    }
