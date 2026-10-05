"""具身层: 把气味场接到 ORN 通道, 把 DN 发放读成轮速, 并跑单个个体的一回合.

建模近似(见 README): 感觉注入为平滑电流(非泊松), DN 池平均发放率直接
解码为差速轮速, 跳过了腹神经索的运动换元.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .brain import BrainConfig, LIFBrain
from . import world as W


def _type_channel(cell_type: str) -> int:
    """把细胞类型稳定地散列到一个 0..9 通道: 0-6 食物, 7-8 毒物, 9 掠食者."""
    h = sum(ord(ch) * (i + 7) for i, ch in enumerate(str(cell_type))) % 10
    return h


@dataclass
class SensorimotorMapping:
    """ORN 通道掩码与 DN 池索引; 全部由真实节点表预先计算."""

    n_nodes: int
    food_l: np.ndarray
    food_r: np.ndarray
    toxin_l: np.ndarray
    toxin_r: np.ndarray
    pred_l: np.ndarray
    pred_r: np.ndarray
    dn_left: np.ndarray
    dn_right: np.ndarray
    dn_center: np.ndarray
    sens_gain: float = 600.0
    speed_gain: float = 0.06
    speed_base: float = 3.0
    speed_max: float = 8.0
    turn_gain: float = 0.04

    @classmethod
    def from_nodes(cls, nodes: pd.DataFrame, **kwargs) -> "SensorimotorMapping":
        orn = nodes[nodes["group"] == "ORN"]
        side = orn["side"].fillna("").str.lower()
        # 未命名的 ORN 用自身 root_id 散列, 保证通道划分稳定且不至于全部挤进一组
        type_key = orn["cell_type"].fillna("")
        type_key = type_key.where(type_key != "", "N" + orn["root_id"].astype(str))
        chan = type_key.map(_type_channel)
        is_food = chan <= 6
        is_toxin = (chan >= 7) & (chan <= 8)
        is_pred = chan == 9
        is_l = side == "left"
        idx = orn["root_id"].index.to_numpy()
        dn = nodes[nodes["group"] == "DN"]
        dn_side = dn["side"].fillna("center").str.lower()
        return cls(
            n_nodes=len(nodes),
            food_l=idx[is_food & is_l],
            food_r=idx[is_food & ~is_l],
            toxin_l=idx[is_toxin & is_l],
            toxin_r=idx[is_toxin & ~is_l],
            pred_l=idx[is_pred & is_l],
            pred_r=idx[is_pred & ~is_l],
            dn_left=dn.index[dn_side == "left"].to_numpy(),
            dn_right=dn.index[dn_side == "right"].to_numpy(),
            dn_center=dn.index[dn_side == "center"].to_numpy(),
            **kwargs,
        )

    def build_ext(self, sens: dict) -> np.ndarray:
        """感觉浓度 -> ORN 外部电流 (左触角注入同侧 ORN)."""
        ext = np.zeros(self.n_nodes)
        for conc_l, conc_r, mask_l, mask_r in [
            (sens["food"][0], sens["food"][1], self.food_l, self.food_r),
            (sens["toxin"][0], sens["toxin"][1], self.toxin_l, self.toxin_r),
            (sens["predator"][0], sens["predator"][1], self.pred_l, self.pred_r),
        ]:
            ext[mask_l] += self.sens_gain * conc_l
            ext[mask_r] += self.sens_gain * conc_r
        return ext

    def wheels_from_counts(self, counts: np.ndarray, steps: int, dt_neural: float) -> tuple[float, float]:
        """DN 池平均发放率(Hz) -> 左右轮速."""
        hz = counts / max(steps * dt_neural, 1e-9)
        rl = float(hz[self.dn_left].mean()) if len(self.dn_left) else 0.0
        rr = float(hz[self.dn_right].mean()) if len(self.dn_right) else 0.0
        speed = float(np.clip(self.speed_base + self.speed_gain * (rl + rr) / 2, 0.0, self.speed_max))
        ang = self.turn_gain * (rr - rl)  # 右强右转
        v_left = speed * (1.0 + ang)
        v_right = speed * (1.0 - ang)
        return v_left, v_right


def evaluate_episode(
    brain: LIFBrain,
    mapping: SensorimotorMapping,
    cfg: W.WorldConfig,
    episode_seed: int,
    noise_seed: int,
    record_every: int = 0,
    track_activity: bool = False,
) -> tuple[dict, list, np.ndarray | None]:
    """跑一个个体的一回合, 返回 (统计字典, 可选轨迹帧, 可选每神经元平均发放Hz)."""
    rng = np.random.default_rng(episode_seed)
    noise_rng = np.random.default_rng(noise_seed)
    s = W.reset_world(cfg, rng, with_predator=cfg.predator_speed > 0)
    # 结构维持成本: 只对"超出基线"的突触计费 (默认关闭, 见 WorldConfig)
    if cfg.metab_per_extra_edge > 0.0:
        s.n_extra_edges = max(0, len(brain.pre) - cfg.n_base_edges)
    state = brain.make_state()
    noise_bank = noise_rng.normal(0.0, brain.cfg.noise_std, size=(256, brain.n))
    steps_per_tick = max(int(round(cfg.dt_ctrl / brain.dt)), 1)
    frames = []
    activity = np.zeros(brain.n) if (track_activity or record_every) else None
    ticks = 0

    while s.agent.alive and s.t < cfg.duration:
        sens = W.sensor_readings(cfg, s)
        ext = mapping.build_ext(sens)
        counts = brain.run_window(state, ext, steps_per_tick, noise_bank)
        v_left, v_right = mapping.wheels_from_counts(counts, steps_per_tick, brain.dt)
        W.move_agent(cfg, s, v_left, v_right, rng)
        W.apply_contacts(cfg, s, rng)
        s.t += cfg.dt_ctrl
        ticks += 1
        if activity is not None:
            activity += counts
        if record_every and (ticks % record_every == 0):
            frames.append(
                (
                    s.agent.pos.copy(),
                    s.agent.heading,
                    s.food_pos.copy(),
                    s.toxin_pos.copy(),
                    s.predator_pos.copy() if s.predator_active else None,
                    s.agent.energy,
                )
            )

    alive_frac = round(min(1.0, s.agent.time_alive / cfg.duration), 3)
    stats = {
        "food": s.food_eaten,
        "toxin_s": round(s.toxin_exposure_s, 2),
        "alive_frac": alive_frac,
        "pred_min_dist": round(float(s.predator_near_min), 1),
        "final_energy": round(float(s.agent.energy), 1),
    }
    if activity is not None:
        activity = activity / max(ticks * steps_per_tick * brain.dt, 1e-9)  # 平均 Hz
    return stats, frames, activity
