# soda/evolution.py
from __future__ import annotations
import numpy as np
import inspect
import hashlib
import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional
from copy import deepcopy

Array = np.ndarray
Primitive = Callable[[Array], Array]


def _hash_grid(x: Array) -> str:
    return hashlib.sha256(x.tobytes()).hexdigest()


def _safe_apply(fn: Primitive, x: Array) -> Optional[Array]:
    try:
        y = fn(x)
        if isinstance(y, np.ndarray) and y.shape == x.shape:
            return y
    except Exception:
        pass
    return None


class PrimitiveGenome:
    def __init__(self, base: Dict[str, Primitive]):
        self.base = base
        self.genes: Dict[str, Primitive] = {}
        self.stats: Dict[str, Dict[str, float]] = {}

    def library(self) -> Dict[str, Primitive]:
        lib = {}
        lib.update(self.base)
        lib.update(self.genes)
        return lib

    def record(self, name: str, success: bool):
        if name not in self.stats:
            self.stats[name] = {"s": 0.0, "t": 0.0}
        self.stats[name]["t"] += 1.0
        if success:
            self.stats[name]["s"] += 1.0

    def weight(self, name: str) -> float:
        s = self.stats.get(name, {"s": 0.0, "t": 0.0})
        return (s["s"] + 1.0) / (s["t"] + 2.0)

    def mutate(self, max_depth: int = 3):
        lib = self.library()
        keys = list(lib.keys())
        if len(keys) < 2:
            return
        k = np.random.randint(2, max_depth + 1)
        seq = list(np.random.choice(keys, size=k, replace=True))
        name = "g_" + hashlib.sha1((".".join(seq)).encode()).hexdigest()[:10]
        if name in lib:
            return

        def make(seq_):
            def f(x: Array) -> Array:
                y = x
                for s in seq_:
                    y2 = _safe_apply(lib[s], y)
                    if y2 is None:
                        return x
                    y = y2
                return y
            return f

        self.genes[name] = make(seq)


@dataclass(order=True)
class Node:
    f: float
    g: float = field(compare=False)
    grid: Array = field(compare=False)
    path: Tuple[str, ...] = field(compare=False)


class SelfEvolvingSolver:
    def __init__(self, primitives: Dict[str, Primitive]):
        self.genome = PrimitiveGenome(primitives)

    def heuristic(self, a: Array, b: Array) -> float:
        return float(np.sum(a != b))

    def solve(
        self,
        start: Array,
        target: Array,
        max_depth: int = 8,
        max_expand: int = 20000,
        evolve_steps: int = 50,
    ) -> Optional[List[str]]:
        for _ in range(evolve_steps):
            self.genome.mutate()

        start = start.copy()
        target = target.copy()

        h0 = self.heuristic(start, target)
        frontier: List[Node] = [Node(h0, 0.0, start, ())]
        visited: Dict[str, float] = {_hash_grid(start): 0.0}
        expand = 0

        while frontier and expand < max_expand:
            node = heapq.heappop(frontier)
            expand += 1

            if np.array_equal(node.grid, target):
                for p in node.path:
                    self.genome.record(p, True)
                return list(node.path)

            if len(node.path) >= max_depth:
                continue

            lib = self.genome.library()
            for name, fn in lib.items():
                y = _safe_apply(fn, node.grid)
                if y is None or np.array_equal(y, node.grid):
                    continue

                g_new = node.g + 1.0 / self.genome.weight(name)
                h_new = self.heuristic(y, target)
                f_new = g_new + h_new
                key = _hash_grid(y)

                if key not in visited or g_new < visited[key]:
                    visited[key] = g_new
                    heapq.heappush(frontier, Node(f_new, g_new, y, node.path + (name,)))
                    self.genome.record(name, False)

        return None


def run_evolution_engine(input_arr: Array, output_arr: Array) -> Optional[List[str]]:
    prims: Dict[str, Primitive] = {
        "rot90": lambda d: np.rot90(d, 1),
        "rot180": lambda d: np.rot90(d, 2),
        "rot270": lambda d: np.rot90(d, 3),
        "flip_h": lambda d: np.fliplr(d),
        "flip_v": lambda d: np.flipud(d),
        "shift_u": lambda d: np.roll(d, -1, axis=0),
        "shift_d": lambda d: np.roll(d, 1, axis=0),
        "shift_l": lambda d: np.roll(d, -1, axis=1),
        "shift_r": lambda d: np.roll(d, 1, axis=1),
    }