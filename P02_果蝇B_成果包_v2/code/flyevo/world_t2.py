"""T2 延迟侧向响应世界 (R1 锚点档, v3): 通道身份线索 -> 3s 静默 -> 转向读出.

把「工作记忆 × 拓扑价值」提纯到计算核心 (经典 delayed-response 范式):
- 线索期   [0, t_cue=2s):   全局 (与位置无关) ORN 通道脉冲 —— 左线索刺激
        食物通道 {0,1,2} 的 ORN, 右线索刺激 {4,5,6} (通道 = 细胞类型哈希,
        与 SensorimotorMapping 同一散列); 代理钉在场心, 朝向自由;
- 延迟期   [t_cue, +3s):    输入全零 —— 记忆必须由回环动力学维持;
- 盲测期   [+3s, +7s):      输入仍为零, 位置钉住, 读净转向 (Σ(v_r−v_l)dt):
        转向符号与线索侧一致且 |净转向| ≥ T0 记满分.

v3 设计史 (如实记录):
- v1 空间版 (气味源 A/B + 盲测导航) 在 probe 上失败 —— fp 宿主是「快速乱撞者」
  (速度贴上限, 靠 30 食物广撒网), 单源趋化与盲测导航都不可达 (ateA 0/8);
- v3 移除全部导航依赖: 线索可靠 (通道身份, 非空间梯度), 读出是运动符号.
  任务耦合两类细接线功能: 通道复用读出 (标签线) + 侧向状态维持 (回环) ——
  block-shuffle 对两者都是合法的破坏 (这就是被检验的假说类).

与 world.py 分工: 复用 WorldConfig/SensorimotorMapping (DN->轮速/转向换算);
感觉注入在本模块 (直接写 ext, 不经 build_ext —— 线索是通道身份不是空间场).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agent import _type_channel
from .world import TWO_PI, AgentState, WorldConfig, WorldState


@dataclass
class T2Config:
    base: WorldConfig
    t_cue: float = 2.0
    t_delay: float = 3.0
    t_test: float = 4.0
    amp: float = 600.0          # 线索脉冲幅度 (与 sens_gain 同尺度)
    turn_sat: float = 1.0       # 净转向饱和阈值 (rad), 预注册
    mirror_cue: bool = False    # probe 用: 交换左右通道 (错配对照)

    @property
    def duration(self) -> float:
        return self.t_cue + self.t_delay + self.t_test


def channel_orn_sets(nodes, mapping) -> tuple[np.ndarray, np.ndarray]:
    """食物通道 ORN 的左右线索子集 (通道哈希: {0,1,2} vs {4,5,6}, 3 留空)."""
    orn = nodes[nodes["group"] == "ORN"]
    type_key = orn["cell_type"].fillna("")
    type_key = type_key.where(type_key != "", "N" + orn["root_id"].astype(str))
    chan = type_key.map(_type_channel)
    idx = orn.index.to_numpy()
    left = idx[chan.isin([0, 1, 2])]
    right = idx[chan.isin([4, 5, 6])]
    return left, right


def reset_t2(cfg: WorldConfig) -> WorldState:
    """代理钉在场心; 侧别由调用方注入 stats (线索即侧别, 无需位置采样)."""
    return WorldState(
        food_pos=np.zeros((0, 2)),
        toxin_pos=np.zeros((0, 2)),
        agent=AgentState(
            pos=np.array([cfg.width / 2, cfg.height / 2]),
            heading=0.0,   # 朝向初值固定 (线索与朝向无关, 固定可减噪声)
            speed=0.0,
            energy=cfg.energy0,
            alive=True,
            time_alive=0.0,
        ),
    )


def run_episode_t2(brain, mapping, t2: T2Config, episode_seed: int, noise_seed: int,
                   cue_side: int | None = None):
    """单回合延迟侧向响应. cue_side: None=按 seed 随机; 0=左通道线索, 1=右通道线索.

    返回 (stats, None, None), 与 world.evaluate_episode 同形.
    """
    base = t2.base
    rng = np.random.default_rng(episode_seed)
    noise_rng = np.random.default_rng(noise_seed)
    s = reset_t2(base)
    if cue_side is None:
        cue_side = int(rng.random() < 0.5)
    if t2.mirror_cue:
        cue_side = 1 - cue_side

    left_set, right_set = t2.cue_sets
    cue_idx = right_set if cue_side == 1 else left_set

    state = brain.make_state()
    noise_bank = noise_rng.normal(0.0, brain.cfg.noise_std, size=(256, brain.n))
    steps_per_tick = max(int(round(base.dt_ctrl / brain.dt)), 1)

    net_turn = 0.0
    while s.agent.alive and s.t < t2.duration:
        ext = np.zeros(brain.n)
        if s.t < t2.t_cue:
            ext[cue_idx] = t2.amp
        # 延迟/盲测期 ext 全零
        counts = brain.run_window(state, ext, steps_per_tick, noise_bank)
        v_left, v_right = mapping.wheels_from_counts(counts, steps_per_tick, brain.dt)
        a = s.agent
        a.speed = max((v_left + v_right) / 2.0, 0.0)
        a.heading += (v_right - v_left) / (2.0 * base.antenna_dist) * base.dt_ctrl
        a.heading += rng.normal(0.0, base.wander_std) * np.sqrt(base.dt_ctrl)
        a.heading = (a.heading + TWO_PI) % TWO_PI
        # 位置钉在场心 (本任务无空间成分), 代谢停计 (无饿死/毒物成分)
        if s.t >= t2.t_cue + t2.t_delay:
            # 盲测窗口: 累计净转向 (右转為正)
            net_turn += (v_right - v_left) / (2.0 * base.antenna_dist) * base.dt_ctrl
        s.t += base.dt_ctrl

    # 记分: 左线索期望左转 (net_turn<0), 右线索期望右转 (>0)
    want = -1.0 if cue_side == 0 else 1.0
    signed = want * net_turn
    match_score = float(np.clip(signed / t2.turn_sat, -1.0, 1.0))
    stats = {
        "food": float(match_score > 0),          # 兼容 "吃到/没吃到" 读数名
        "match_score": round(match_score, 4),
        "net_turn": round(float(net_turn), 4),
        "side_left": float(cue_side == 0),
        "toxin_s": 0.0,
        "alive_frac": 1.0,
    }
    return stats, None, None


def fitness_t2(st: dict) -> float:
    """10 × 连续记分 (方向匹配 × 幅度饱和) —— 机会水平 = 0."""
    return 10.0 * st["match_score"]


def make_pinned_config(base_world: WorldConfig) -> WorldConfig:
    """T2 专用物理: wander 0.5 (probe 校准后冻结)."""
    from dataclasses import replace
    return replace(base_world, wander_std=0.5)
