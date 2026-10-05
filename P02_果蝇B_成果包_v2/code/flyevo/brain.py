"""稀疏 LIF 脑网络: 拓扑固定(真实连接组), 突触权重可进化.

不用 scipy: 每步用 np.bincount 沿边表传递发放, 33k 边规模下单步 ~60µs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BrainConfig:
    dt: float = 2e-3          # 神经步长 (s)
    v_rest: float = -60.0
    v_thresh: float = -50.0
    v_reset: float = -70.0
    tau_mem: float = 0.02     # 膜时间常数 (s)
    tau_syn: float = 0.005    # 突触电流衰减 (s)
    refr_steps: int = 1       # 不应期步数
    syn_gain: float = 12000.0  # 每突触事件注入的电流标度 (标定: ORN->PN 传得过阈)
    noise_std: float = 20.0   # 背景电流噪声 (每神经步)


class LIFBrain:
    """一群 LIF 神经元 + 固定拓扑的有符号突触。

    权重向量 w 与 sign 相乘后作为边强度; 约束 w >= w_min, 即进化只调
    强度不能翻转递质极性(GABA 永远抑制) -- 这是生物学上的合理约束.

    `w0` 是**固定基权重** (不参与进化), 用来承载解剖学先验:
      - w0 ≡ 1 (默认)      -> v1 行为, 逐位可复现
      - w0 = syn_count 归一 -> 与 sign 相乘后, 同一神经元对的所有行求和 =
        Σ sign(nt)×syn_count = N_exc − N_inh, 即 FlyGM eq.1 的净极化突触计数。
    进化基因组 w 在其上做乘性缩放, 因此 w0 只改变"起点与相对权重", 不改变可塑性范围。
    """

    def __init__(
        self,
        n_nodes: int,
        pre_idx: np.ndarray,
        post_idx: np.ndarray,
        sign: np.ndarray,
        cfg: BrainConfig | None = None,
        w0: np.ndarray | None = None,
    ):
        self.n = n_nodes
        self.pre = pre_idx
        self.post = post_idx
        self.sign = sign.astype(np.float64)
        self.w0 = (np.ones_like(self.sign) if w0 is None
                   else np.asarray(w0, dtype=np.float64))
        if self.w0.shape != self.sign.shape:
            raise ValueError(f"w0 形状 {self.w0.shape} != sign 形状 {self.sign.shape}")
        self.cfg = cfg or BrainConfig()
        self.dt = self.cfg.dt
        self._syn_decay = float(np.exp(-self.dt / self.cfg.tau_syn))
        self._mem_decay = float(np.exp(-self.dt / self.cfg.tau_mem))
        self.reset_weights()

    def reset_weights(self) -> None:
        self.w = np.ones_like(self.sign)

    def set_weights(self, w: np.ndarray) -> None:
        self.w = w

    def make_state(self) -> dict:
        return {
            "v": np.full(self.n, self.cfg.v_rest),
            "i_syn": np.zeros(self.n),
            "refr": np.zeros(self.n, dtype=np.int32),
        }

    def step(
        self,
        state: dict,
        ext_current: np.ndarray,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """推进一步, 返回本步发放的布尔向量."""
        c = self.cfg
        v = state["v"]
        v *= self._mem_decay
        v += (1.0 - self._mem_decay) * c.v_rest
        v += self.dt * (state["i_syn"] + ext_current)
        if noise is not None:
            v += self.dt * noise

        can_spike = state["refr"] <= 0
        spiked = (v >= c.v_thresh) & can_spike
        v[spiked] = c.v_reset
        state["refr"][spiked] = c.refr_steps
        state["refr"] -= 1

        state["i_syn"] *= self._syn_decay
        # w0 承载解剖学基权重 (syn_count 口径时, 逐对求和 = N_exc − N_inh)
        delivered = self.w * self.w0 * spiked[self.pre] * self.sign
        state["i_syn"] += np.bincount(
            self.post, weights=delivered, minlength=self.n
        ) * c.syn_gain
        return spiked

    def run_window(
        self,
        state: dict,
        ext_current: np.ndarray,
        steps: int,
        noise_bank: np.ndarray | None = None,
    ) -> np.ndarray:
        """跑一个控制窗 (steps 个神经步), 返回每神经元累计发放数."""
        counts = np.zeros(self.n)
        for k in range(steps):
            noise = None
            if noise_bank is not None:
                noise = noise_bank[k % len(noise_bank)]
            counts += self.step(state, ext_current, noise)
        return counts
