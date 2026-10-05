"""桥3: 亚群隔离 → 方言分化 → 混血/注入传承.

对齐 GlossoGen swap 与动物方言: 发明难, 继承可测.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Population:
    name: str
    genomes: np.ndarray
    masks: np.ndarray | None = None
    fitness: np.ndarray | None = None
    gen: int = 0
    history: list = field(default_factory=list)


def split_population(
    base_genomes: np.ndarray, n_a: int, n_b: int, rng: np.random.Generator
):
    n = base_genomes.shape[0]
    idx = rng.permutation(n)
    ga = base_genomes[idx[:n_a]].copy()
    gb = base_genomes[idx[n_a:n_a + n_b]].copy()
    ga *= rng.lognormal(0, 0.1, size=ga.shape)
    gb *= rng.lognormal(0, 0.1, size=gb.shape)
    return Population("A", ga), Population("B", gb)


def fixed_edge_set_from_masks(
    masks: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    real_size: int,
    thresh: float = 0.95,
):
    if masks is None:
        return set()
    freq = masks[:, real_size:].mean(axis=0)
    ids = np.flatnonzero(freq >= thresh)
    out = set()
    for j in ids:
        e = real_size + int(j)
        if e < len(pre):
            out.add((int(pre[e]), int(post[e])))
    return out


def hybridize(
    a: Population, b: Population, n_child: int, rng: np.random.Generator
) -> Population:
    ga, gb = a.genomes, b.genomes
    n_a = ga.shape[0]
    n_b = gb.shape[0]
    d = ga.shape[1]
    children = []
    for _ in range(n_child):
        ia, ib = rng.integers(0, n_a), rng.integers(0, n_b)
        mask = rng.random(d) < 0.5
        children.append(np.where(mask, ga[ia], gb[ib]))
    genomes = np.array(children)
    masks = None
    if a.masks is not None and b.masks is not None:
        cap = a.masks.shape[1]
        cm = []
        for _ in range(n_child):
            ia, ib = rng.integers(0, n_a), rng.integers(0, n_b)
            m = rng.random(cap) < 0.5
            cm.append(np.where(m, a.masks[ia], b.masks[ib]))
        masks = np.array(cm)
    return Population("Hybrid", genomes, masks=masks, gen=max(a.gen, b.gen))


def inject_best(
    host: Population,
    donor_best: np.ndarray,
    frac: float = 0.1,
    noise: float = 0.02,
    rng: np.random.Generator | None = None,
) -> Population:
    rng = rng or np.random.default_rng(0)
    g = host.genomes.copy()
    n = len(g)
    k = max(1, int(round(frac * n)))
    idx = rng.choice(n, size=k, replace=False)
    g[idx] = donor_best[None, :] * rng.lognormal(
        0, noise, size=(k, donor_best.shape[0])
    )
    return Population(
        host.name + "+inj",
        g,
        masks=host.masks.copy() if host.masks is not None else None,
        gen=host.gen,
    )


def fresh_population(n: int, w_init: np.ndarray, rng: np.random.Generator) -> Population:
    g = w_init[None, :] * rng.lognormal(0, 0.10, size=(n, len(w_init)))
    return Population("Fresh", g)


def best_genome(pop: Population) -> np.ndarray:
    if pop.fitness is None:
        return pop.genomes[int(np.argmax(pop.genomes.mean(axis=1)))]
    return pop.genomes[int(np.argmax(pop.fitness))]
