# soda/evolution.py
Self-Evolving Solver for ARC-like Grid Reasoning
=================================================
Architecture:
  PrimitiveRegistry   → frozen, typed, composable primitives
  PrimitiveGenome     → adaptive library with Bayesian weights + mutation
  SelfEvolvingSolver  → A* search over genome with shape-aware heuristic
  KnowledgeBase       → persistent cross-task memory (singleton)

Design Invariants:
  - Every primitive: pure function, no side-effects
  - Shape contract: explicitly declared (same | any | square)
  - Search: weighted A* with inadmissible inflation for speed/quality tradeoff
  - Memory: stats survive across tasks via KnowledgeBase singleton
"""
from __future__ import annotations

import hashlib
import heapq
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, Final, Iterator, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

Array  = np.ndarray
PrimFn = Callable[[Array], Optional[Array]]

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shape Contract
# ─────────────────────────────────────────────────────────────────────────────

class ShapeContract(Enum):
    """Declares what a primitive guarantees about output shape."""
    SAME   = auto()   # output.shape == input.shape  (enforced)
    ANY    = auto()   # output.shape is unconstrained (allowed)
    SQUARE = auto()   # valid only on square grids    (guarded)


# ─────────────────────────────────────────────────────────────────────────────
# Primitive Descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PrimitiveDescriptor:
    """Immutable metadata attached to every registered primitive."""
    name:     str
    fn:       PrimFn
    contract: ShapeContract
    group:    str

    def __call__(self, x: Array) -> Optional[Array]:
        try:
            if self.contract is ShapeContract.SQUARE and x.shape[0] != x.shape[1]:
                return None
            result = self.fn(x)
            if result is None or not isinstance(result, np.ndarray):
                return None
            if self.contract is ShapeContract.SAME and result.shape != x.shape:
                return None
            return result
        except Exception:
            return None

    # Required for frozen dataclass with callable field
    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PrimitiveDescriptor) and self.name == other.name


# ─────────────────────────────────────────────────────────────────────────────
# Color Helpers  (pure, zero side-effects)
# ─────────────────────────────────────────────────────────────────────────────

def _recolor(grid: Array, src: int, dst: int) -> Array:
    out = grid.copy()
    out[grid == src] = dst
    return out


def _mask_color(grid: Array, color: int) -> Array:
    out = np.zeros_like(grid)
    out[grid == color] = color
    return out


def _flood_background(grid: Array, color: int) -> Array:
    out = grid.copy()
    out[grid == 0] = color
    return out


def _swap_colors(grid: Array, a: int, b: int) -> Array:
    out = grid.copy()
    out[grid == a] = b
    out[grid == b] = a
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Primitive Registry
# ─────────────────────────────────────────────────────────────────────────────

class PrimitiveRegistry:
    """
    Frozen, append-only registry of atomic primitives.
    Build the default set with `PrimitiveRegistry.build()`.
    """

    def __init__(self) -> None:
        self._store: Dict[str, PrimitiveDescriptor] = {}

    def register(self, desc: PrimitiveDescriptor) -> None:
        if desc.name in self._store:
            raise KeyError(f"Primitive '{desc.name}' already registered.")
        self._store[desc.name] = desc

    def get(self, name: str) -> Optional[PrimitiveDescriptor]:
        return self._store.get(name)

    def all(self) -> Iterator[PrimitiveDescriptor]:
        return iter(self._store.values())

    def as_dict(self) -> Dict[str, PrimitiveDescriptor]:
        return dict(self._store)

    def __len__(self) -> int:
        return len(self._store)

    # ── Default Factory ───────────────────────────────────────────────────────

    @classmethod
    def build(cls) -> "PrimitiveRegistry":
        """Build and return the complete default primitive set."""
        reg = cls()
        ARC_COLORS: Final[range] = range(10)

        # ── Geometric ────────────────────────────────────────────────────────
        for k, name in [(1, "rot90"), (2, "rot180"), (3, "rot270")]:
            _k = k
            reg.register(PrimitiveDescriptor(
                name=name,
                fn=lambda d, k=_k: np.rot90(d, k),
                contract=ShapeContract.SQUARE,
                group="rotation",
            ))

        reg.register(PrimitiveDescriptor(
            name="flip_h",
            fn=lambda d: np.fliplr(d),
            contract=ShapeContract.SAME,
            group="flip",
        ))
        reg.register(PrimitiveDescriptor(
            name="flip_v",
            fn=lambda d: np.flipud(d),
            contract=ShapeContract.SAME,
            group="flip",
        ))
        reg.register(PrimitiveDescriptor(
            name="transpose",
            fn=lambda d: d.T.copy(),
            contract=ShapeContract.SQUARE,
            group="flip",
        ))

        # ── Shift (toroidal) ─────────────────────────────────────────────────
        for axis, amount, name in [
            (0, -1, "shift_u"), (0, 1, "shift_d"),
            (1, -1, "shift_l"), (1, 1, "shift_r"),
        ]:
            _a, _m = axis, amount
            reg.register(PrimitiveDescriptor(
                name=name,
                fn=lambda d, a=_a, m=_m: np.roll(d, m, axis=a),
                contract=ShapeContract.SAME,
                group="shift",
            ))

        # ── Color Operations ─────────────────────────────────────────────────
        for src in ARC_COLORS:
            for dst in ARC_COLORS:
                if src == dst:
                    continue
                _s, _d = src, dst
                reg.register(PrimitiveDescriptor(
                    name=f"recolor_{_s}_{_d}",
                    fn=lambda d, s=_s, t=_d: _recolor(d, s, t),
                    contract=ShapeContract.SAME,
                    group="color",
                ))

        for c in ARC_COLORS:
            _c = c
            reg.register(PrimitiveDescriptor(
                name=f"mask_{_c}",
                fn=lambda d, c=_c: _mask_color(d, c),
                contract=ShapeContract.SAME,
                group="color",
            ))
            reg.register(PrimitiveDescriptor(
                name=f"flood_{_c}",
                fn=lambda d, c=_c: _flood_background(d, c),
                contract=ShapeContract.SAME,
                group="color",
            ))

        reg.register(PrimitiveDescriptor(
            name="invert_colors",
            fn=lambda d: (9 - d).clip(0, 9),
            contract=ShapeContract.SAME,
            group="color",
        ))
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            _a, _b = a, b
            reg.register(PrimitiveDescriptor(
                name=f"swap_{_a}_{_b}",
                fn=lambda d, a=_a, b=_b: _swap_colors(d, a, b),
                contract=ShapeContract.SAME,
                group="color",
            ))

        # ── Scale / Tile ──────────────────────────────────────────────────────
        for f in [2, 3]:
            _f = f
            reg.register(PrimitiveDescriptor(
                name=f"scale_up_{_f}x",
                fn=lambda d, f=_f: np.repeat(np.repeat(d, f, axis=0), f, axis=1),
                contract=ShapeContract.ANY,
                group="scale",
            ))

        reg.register(PrimitiveDescriptor(
            name="tile_2x2",
            fn=lambda d: np.tile(d, (2, 2)),
            contract=ShapeContract.ANY,
            group="scale",
        ))
        reg.register(PrimitiveDescriptor(
            name="trim_border",
            fn=lambda d: d[1:-1, 1:-1] if d.shape[0] > 2 and d.shape[1] > 2 else d,
            contract=ShapeContract.ANY,
            group="scale",
        ))

        # ── Logical ───────────────────────────────────────────────────────────
        reg.register(PrimitiveDescriptor(
            name="binarize",
            fn=lambda d: (d > 0).astype(d.dtype),
            contract=ShapeContract.SAME,
            group="logical",
        ))
        reg.register(PrimitiveDescriptor(
            name="identity",
            fn=lambda d: d.copy(),
            contract=ShapeContract.SAME,
            group="identity",
        ))

        return reg


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base  (cross-task persistent memory, singleton)
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Singleton persistent memory for primitive statistics across tasks.
    Stats survive solver instantiation; serialisable to/from JSON.
    """

    _instance: Optional["KnowledgeBase"] = None

    def __new__(cls) -> "KnowledgeBase":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._stats: Dict[str, Dict[str, float]] = {}
            cls._instance = obj
        return cls._instance

    def record(self, name: str, success: bool) -> None:
        if name not in self._stats:
            self._stats[name] = {"s": 0.0, "t": 0.0}
        self._stats[name]["t"] += 1.0
        if success:
            self._stats[name]["s"] += 1.0

    def weight(self, name: str) -> float:
        """Laplace-smoothed success rate in (0, 1)."""
        s = self._stats.get(name, {"s": 0.0, "t": 0.0})
        return (s["s"] + 1.0) / (s["t"] + 2.0)

    def top_k(self, k: int = 10) -> List[Tuple[str, float]]:
        return sorted(
            [(n, self.weight(n)) for n in self._stats],
            key=lambda x: x[1],
            reverse=True,
        )[:k]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self._stats, indent=2))

    def load(self, path: Path) -> None:
        if path.exists():
            self._stats = json.loads(path.read_text())

    def reset(self) -> None:
        self._stats.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Primitive Genome  (mutable evolved library)
# ─────────────────────────────────────────────────────────────────────────────

class PrimitiveGenome:
    """
    Mutable library of composed (gene) primitives on top of a frozen
    base registry. Mutation creates new depth-k weighted compositions.
    """

    MAX_GENES: Final[int] = 512

    def __init__(self, registry: PrimitiveRegistry, kb: KnowledgeBase) -> None:
        self._registry = registry
        self._kb       = kb
        self._genes:   Dict[str, PrimitiveDescriptor] = {}

    def library(self) -> Dict[str, PrimitiveDescriptor]:
        return {**self._registry.as_dict(), **self._genes}

    def weight(self, name: str) -> float:
        return self._kb.weight(name)

    def record(self, name: str, success: bool) -> None:
        self._kb.record(name, success)

    def mutate(
        self,
        max_depth: int = 3,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Create one gene by composing k weighted-sampled primitives."""
        if len(self._genes) >= self.MAX_GENES:
            return
        if rng is None:
            rng = np.random.default_rng()

        lib   = self.library()
        keys  = list(lib.keys())
        if len(keys) < 2:
            return

        weights = np.array([self._kb.weight(k) for k in keys], dtype=float)
        weights /= weights.sum()

        depth    = int(rng.integers(2, max_depth + 1))
        selected = rng.choice(keys, size=depth, replace=True, p=weights).tolist()
        gene_id  = "g_" + hashlib.sha1("|".join(selected).encode()).hexdigest()[:12]

        if gene_id in lib:
            return

        _snapshot = {k: lib[k] for k in selected if k in lib}

        def _composed(
            x:   Array,
            seq: List[str]                      = selected,
            snp: Dict[str, PrimitiveDescriptor] = _snapshot,
        ) -> Optional[Array]:
            y = x
            for s in seq:
                if s not in snp:
                    return None
                result = snp[s](y)
                if result is None:
                    return y
                y = result
            return y

        self._genes[gene_id] = PrimitiveDescriptor(
            name=gene_id,
            fn=_composed,
            contract=ShapeContract.ANY,
            group="gene",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Search Internals
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic(a: Array, b: Array, eps: float = 1.0) -> float:
    """
    Shape-aware heuristic for weighted A*.
    Returns cell-wise mismatch when shapes match.
    Returns size-penalised estimate when shapes differ.
    """
    if a.shape == b.shape:
        return float(np.sum(a != b)) * eps
    size_diff = abs(a.size - b.size)
    return float(max(a.size, b.size) + size_diff * 2) * eps


def _hash(x: Array) -> str:
    return hashlib.sha256(x.tobytes()).hexdigest()


@dataclass(order=True)
class _Node:
    f:    float
    g:    float           = field(compare=False)
    grid: Array           = field(compare=False)
    path: Tuple[str, ...] = field(compare=False)


# ─────────────────────────────────────────────────────────────────────────────
# Self-Evolving Solver
# ─────────────────────────────────────────────────────────────────────────────

class SelfEvolvingSolver:
    """
    Weighted A* solver over a self-evolving primitive genome.

    Parameters
    ----------
    registry   : PrimitiveRegistry  — frozen base set
    kb         : KnowledgeBase      — persistent cross-task stats
    weight_eps : float              — A* inflation  (1.0 = optimal, >1 = faster)
    """

    def __init__(
        self,
        registry:   PrimitiveRegistry,
        kb:         KnowledgeBase,
        weight_eps: float = 1.5,
    ) -> None:
        self._genome     = PrimitiveGenome(registry, kb)
        self._weight_eps = weight_eps

    def solve(
        self,
        start:        Array,
        target:       Array,
        max_depth:    int   = 8,
        max_expand:   int   = 25_000,
        evolve_steps: int   = 60,
        rng:          Optional[np.random.Generator] = None,
    ) -> Optional[List[str]]:
        """
        Search for a primitive sequence transforming `start` → `target`.

        Returns
        -------
        List[str] | None
            Ordered primitive names, or None if budget exceeded.
        """
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(evolve_steps):
            self._genome.mutate(rng=rng)

        start  = start.copy()
        target = target.copy()

        if np.array_equal(start, target):
            return []

        h0       = _heuristic(start, target, self._weight_eps)
        frontier: List[_Node]         = [_Node(h0, 0.0, start, ())]
        visited:  Dict[str, float]    = {_hash(start): 0.0}
        expand   = 0

        while frontier and expand < max_expand:
            node = heapq.heappop(frontier)
            expand += 1

            if np.array_equal(node.grid, target):
                for p in node.path:
                    self._genome.record(p, True)
                logger.debug("Solved | expansions=%d path=%s", expand, node.path)
                return list(node.path)

            if len(node.path) >= max_depth:
                continue

            for name, prim in self._genome.library().items():
                y = prim(node.grid)
                if y is None or np.array_equal(y, node.grid):
                    continue

                cost  = 1.0 / self._genome.weight(name)
                g_new = node.g + cost
                h_new = _heuristic(y, target, self._weight_eps)
                f_new = g_new + h_new
                key   = _hash(y)

                if key not in visited or g_new < visited[key]:
                    visited[key] = g_new
                    heapq.heappush(frontier, _Node(f_new, g_new, y, node.path + (name,)))
                    self._genome.record(name, False)

        logger.debug("No solution | expansions=%d", expand)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Module-level API
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_REGISTRY: Optional[PrimitiveRegistry] = None
_DEFAULT_KB:       KnowledgeBase               = KnowledgeBase()


def _get_registry() -> PrimitiveRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = PrimitiveRegistry.build()
    return _DEFAULT_REGISTRY


def run_evolution_engine(
    input_arr:    Array,
    output_arr:   Array,
    max_depth:    int   = 8,
    max_expand:   int   = 25_000,
    evolve_steps: int   = 60,
    weight_eps:   float = 1.5,
    seed:         Optional[int] = None,
) -> Optional[List[str]]:
    """
    Convenience wrapper — uses the module-level persistent KnowledgeBase.

    Parameters
    ----------
    input_arr    : 2D numpy int array  (ARC input  grid)
    output_arr   : 2D numpy int array  (ARC target grid)
    max_depth    : maximum composition length
    max_expand   : A* node expansion budget
    evolve_steps : genome warm-up mutation count
    weight_eps   : inadmissible A* inflation factor
    seed         : RNG seed for reproducibility

    Returns
    -------
    List[str] | None
    """
    registry = _get_registry()
    rng      = np.random.default_rng(seed)
    solver   = SelfEvolvingSolver(registry, _DEFAULT_KB, weight_eps=weight_eps)
    return solver.solve(
        start=input_arr,
        target=output_arr,
        max_depth=max_depth,
        max_expand=max_expand,
        evolve_steps=evolve_steps,
        rng=rng,
    )
