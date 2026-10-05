"""桥2: 多体评估 — 每只果蝇一套脑; 信号由 DN 池解码.

发射编码(最小可用):
  - DN 中心池平均发放率 → 释放量
  - 左右 DN 差 → 通道 (0 食物标记 / 1 警戒)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .brain import LIFBrain
from .agent import SensorimotorMapping
from .social_world import (
    SocialWorldConfig,
    reset_social,
    social_sensor_readings,
    step_social,
    social_stats,
)


@dataclass
class SignalCodec:
    rate_to_release: float = 0.02
    release_threshold_hz: float = 20.0
    channel_lr_bias: float = 0.0


def build_social_ext(
    mapping: SensorimotorMapping,
    sens: dict,
    pher_gain: float,
) -> np.ndarray:
    ext = np.zeros(mapping.n_nodes)
    for conc_l, conc_r, mask_l, mask_r in [
        (sens["food"][0], sens["food"][1], mapping.food_l, mapping.food_r),
        (sens["toxin"][0], sens["toxin"][1], mapping.toxin_l, mapping.toxin_r),
        (sens["predator"][0], sens["predator"][1], mapping.pred_l, mapping.pred_r),
    ]:
        ext[mask_l] += mapping.sens_gain * conc_l
        ext[mask_r] += mapping.sens_gain * conc_r
    if len(mapping.food_l):
        ext[mapping.food_l] += pher_gain * sens["sig_food"][0]
        ext[mapping.food_r] += pher_gain * sens["sig_food"][1]
    if len(mapping.toxin_l):
        ext[mapping.toxin_l] += pher_gain * sens["sig_alarm"][0]
        ext[mapping.toxin_r] += pher_gain * sens["sig_alarm"][1]
    return ext


def decode_signal(
    counts: np.ndarray,
    steps: int,
    dt_neural: float,
    mapping: SensorimotorMapping,
    codec: SignalCodec,
):
    hz = counts / max(steps * dt_neural, 1e-9)
    if len(mapping.dn_center):
        drive = float(hz[mapping.dn_center].mean())
    else:
        drive = float(
            (hz[mapping.dn_left].mean() if len(mapping.dn_left) else 0.0)
            + (hz[mapping.dn_right].mean() if len(mapping.dn_right) else 0.0)
        ) / 2
    if drive < codec.release_threshold_hz:
        return 0.0, 0
    release = float(min(1.0, codec.rate_to_release * drive))
    rl = float(hz[mapping.dn_left].mean()) if len(mapping.dn_left) else 0.0
    rr = float(hz[mapping.dn_right].mean()) if len(mapping.dn_right) else 0.0
    ch = 1 if (rr - rl) > codec.channel_lr_bias else 0
    return release, ch


def evaluate_social_episode(
    brains: list,
    mapping: SensorimotorMapping,
    cfg: SocialWorldConfig,
    episode_seed: int,
    noise_seed: int,
    codec: SignalCodec | None = None,
    record_every: int = 0,
):
    codec = codec or SignalCodec()
    rng = np.random.default_rng(episode_seed)
    nrng = np.random.default_rng(noise_seed)
    s = reset_social(cfg, rng, with_predator=cfg.predator_speed > 0)
    states = [b.make_state() for b in brains]
    noise_banks = [
        nrng.normal(0.0, b.cfg.noise_std, size=(256, b.n)) for b in brains
    ]
    dt_n = brains[0].cfg.dt if brains else 0.002
    n_ctrl = max(1, int(round(cfg.duration / cfg.dt_ctrl)))
    neural_per_ctrl = max(1, int(round(cfg.dt_ctrl / dt_n)))
    frames = [] if record_every else None

    for step in range(n_ctrl):
        wheels, releases = [], []
        for i, brain in enumerate(brains):
            if not s.flies[i].alive:
                wheels.append((0.0, 0.0))
                releases.append((0.0, 0))
                continue
            sens = social_sensor_readings(cfg, s, i)
            ext = build_social_ext(mapping, sens, cfg.pheromone_gain)
            counts = brain.run_window(
                states[i], ext, neural_per_ctrl, noise_banks[i]
            )
            wheels.append(mapping.wheels_from_counts(counts, neural_per_ctrl, dt_n))
            releases.append(decode_signal(counts, neural_per_ctrl, dt_n, mapping, codec))
        s = step_social(cfg, s, wheels, releases, rng)
        if record_every and step % record_every == 0:
            frames.append(
                {
                    "t": s.t,
                    "flies": [(float(f.pos[0]), float(f.pos[1]), f.alive, f.release) for f in s.flies],
                    "food": s.food_pos.copy(),
                }
            )

    stats = social_stats(s)
    return stats, frames
