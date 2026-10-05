"""演化剧本: 按生物演化进程分阶段施加选择压力.

种群跨阶段延续 -- 这是"累积演化"的关键设定: 阶段 1 进化出的觅食线路,
到阶段 3 要在掠食者存在下继续工作.
"""
from __future__ import annotations

from dataclasses import replace

from . import world as W


def fitness_s1(st: dict) -> float:
    """阶段1 代谢: 吃到就是硬道理."""
    return 10.0 * st["food"] + 20.0 * st["alive_frac"]


def fitness_s2(st: dict) -> float:
    """阶段2 毒性: 觅食成绩被毒物暴露打折."""
    return 10.0 * st["food"] + 20.0 * st["alive_frac"] - 8.0 * st["toxin_s"]


def fitness_s34(st: dict) -> float:
    """阶段3/4 掠食: 活下来权重最大."""
    return 10.0 * st["food"] + 60.0 * st["alive_frac"] - 8.0 * st["toxin_s"]


def _stage(idx: int, name: str, desc: str, fitness, **world_overrides) -> dict:
    base = W.WorldConfig()
    return {
        "idx": idx,
        "name": name,
        "desc": desc,
        "fitness": fitness,
        "world": replace(base, **world_overrides),
    }


STAGES = [
    _stage(
        1,
        "代谢",
        "只有食物。选择压力: 获能效率 -> 进化出趋化取食反射.",
        fitness_s1,
        predator_speed=0.0,
    ),
    _stage(
        2,
        "毒性",
        "加入毒物。吃到毒物持续掉血 -> 进化气味回避.",
        fitness_s2,
        predator_speed=0.0,
    ),
    _stage(
        3,
        "掠食",
        "加入掠食者(追踪者)。接触即死 -> 进化逃逸反射.",
        fitness_s34,
        predator_speed=3.0,
    ),
    _stage(
        4,
        "剧变",
        "掠食者提速50%, 食物减半。环境突变 -> 适应或灭绝.",
        fitness_s34,
        predator_speed=4.5,
        n_food=12,
    ),
]


def get_stages(which: list[int] | None) -> list[dict]:
    if not which:
        return STAGES
    return [s for s in STAGES if s["idx"] in which]
