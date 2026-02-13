# soda/tests/test_evolution.py
Full benchmark suite for evolution.py
======================================
Groups:
  Geometric   - rotation, flip, transpose
  Shift       - toroidal shift all directions
  Composition - multi-step paths, A* optimality check
  Color       - recolor, mask, flood, invert, swap
  Scale       - scale_up, tile_2x2
  Boundary    - impossible tasks, shape mismatch, crash guard
  Memory      - KnowledgeBase persistence across solver instances
  Performance - timeout smoke tests
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from soda.evolution import (
    KnowledgeBase,
    PrimitiveRegistry,
    SelfEvolvingSolver,
    _heuristic,
    run_evolution_engine,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_kb():
    KnowledgeBase().reset()
    yield
    KnowledgeBase().reset()


@pytest.fixture
def registry():
    return PrimitiveRegistry.build()


@pytest.fixture
def kb():
    return KnowledgeBase()


@pytest.fixture
def solver(registry, kb):
    return SelfEvolvingSolver(registry, kb, weight_eps=1.5)


def solve(inp, out, seed=42, **kwargs):
    return run_evolution_engine(np.array(inp), np.array(out), seed=seed, **kwargs)


def verify(solution: Optional[List[str]], inp, out) -> bool:
    if solution is None:
        return False
    reg  = PrimitiveRegistry.build()
    grid = np.array(inp).copy()
    for step in solution:
        prim = reg.get(step)
        if prim is None:
            return False
        result = prim(grid)
        if result is None:
            return False
        grid = result
    return np.array_equal(grid, np.array(out))


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic
# ─────────────────────────────────────────────────────────────────────────────

class TestHeuristic:
    def test_identical_zero(self):
        a = np.array([[1, 2], [3, 4]])
        assert _heuristic(a, a) == 0.0

    def test_one_cell_diff(self):
        a = np.array([[1, 0], [0, 0]])
        b = np.array([[2, 0], [0, 0]])
        assert _heuristic(a, b) == 1.0

    def test_shape_mismatch_finite(self):
        a = np.zeros((2, 2), dtype=int)
        b = np.zeros((4, 4), dtype=int)
        assert np.isfinite(_heuristic(a, b))

    def test_shape_mismatch_positive(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert _heuristic(a, b) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_geometric_exists(self, registry):
        for n in ["rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"]:
            assert registry.get(n) is not None, n

    def test_shift_exists(self, registry):
        for n in ["shift_u", "shift_d", "shift_l", "shift_r"]:
            assert registry.get(n) is not None, n

    def test_color_exists(self, registry):
        assert registry.get("recolor_1_2") is not None
        assert registry.get("mask_0")      is not None
        assert registry.get("flood_3")     is not None
        assert registry.get("invert_colors") is not None

    def test_scale_exists(self, registry):
        assert registry.get("scale_up_2x") is not None
        assert registry.get("tile_2x2")    is not None

    def test_duplicate_raises(self, registry):
        desc = registry.get("flip_h")
        with pytest.raises(KeyError):
            registry.register(desc)

    def test_count_sufficient(self, registry):
        assert len(registry) >= 130


# ─────────────────────────────────────────────────────────────────────────────
# Geometric
# ─────────────────────────────────────────────────────────────────────────────

class TestGeometric:
    G = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_trivial(self):
        assert solve([[1, 2], [3, 4]], [[1, 2], [3, 4]]) == []

    def test_flip_h(self):
        out = [[3,2,1],[6,5,4],[9,8,7]]
        s = solve(self.G, out)
        assert s is not None and verify(s, self.G, out)

    def test_flip_v(self):
        out = [[7,8,9],[4,5,6],[1,2,3]]
        s = solve(self.G, out)
        assert s is not None and verify(s, self.G, out)

    def test_rot90(self):
        s = solve([[1,2],[3,4]], [[2,4],[1,3]])
        assert s is not None and verify(s, [[1,2],[3,4]], [[2,4],[1,3]])

    def test_rot180(self):
        out = [[9,8,7],[6,5,4],[3,2,1]]
        s = solve(self.G, out)
        assert s is not None and verify(s, self.G, out)

    def test_rot270(self):
        s = solve([[1,2],[3,4]], [[3,1],[4,2]])
        assert s is not None and verify(s, [[1,2],[3,4]], [[3,1],[4,2]])

    def test_transpose(self):
        out = [[1,4,7],[2,5,8],[3,6,9]]
        s = solve(self.G, out)
        assert s is not None and verify(s, self.G, out)


# ─────────────────────────────────────────────────────────────────────────────
# Shift
# ─────────────────────────────────────────────────────────────────────────────

class TestShift:
    def test_shift_r(self):
        s = solve([[0,1,0],[0,1,0],[0,1,0]], [[0,0,1],[0,0,1],[0,0,1]])
        assert s is not None and verify(s, [[0,1,0],[0,1,0],[0,1,0]], [[0,0,1],[0,0,1],[0,0,1]])

    def test_shift_l(self):
        s = solve([[0,0,1],[0,0,1],[0,0,1]], [[0,1,0],[0,1,0],[0,1,0]])
        assert s is not None

    def test_shift_u(self):
        s = solve([[0,0,0],[1,1,1],[0,0,0]], [[1,1,1],[0,0,0],[0,0,0]])
        assert s is not None

    def test_shift_d(self):
        s = solve([[1,1,1],[0,0,0],[0,0,0]], [[0,0,0],[1,1,1],[0,0,0]])
        assert s is not None


# ─────────────────────────────────────────────────────────────────────────────
# Composition
# ─────────────────────────────────────────────────────────────────────────────

class TestComposition:
    def test_flip_h_flip_v_eq_rot180(self):
        inp = [[1,0,0],[0,0,0],[0,0,2]]
        out = [[2,0,0],[0,0,0],[0,0,1]]
        s = solve(inp, out)
        assert s is not None and verify(s, inp, out)

    def test_two_step(self):
        inp = [[1,2,3],[4,5,6],[7,8,9]]
        out = [[9,6,3],[8,5,2],[7,4,1]]
        s = solve(inp, out)
        assert s is not None and verify(s, inp, out)

    def test_astar_prefers_short_path(self):
        inp = [[1,0],[0,2]]
        out = [[2,0],[0,1]]
        s = solve(inp, out)
        assert s is not None and len(s) <= 2


# ─────────────────────────────────────────────────────────────────────────────
# Color
# ─────────────────────────────────────────────────────────────────────────────

class TestColor:
    def test_recolor_1_to_2(self):
        inp = [[1,0,0],[0,1,0],[0,0,1]]
        out = [[2,0,0],[0,2,0],[0,0,2]]
        s = solve(inp, out)
        assert s is not None and verify(s, inp, out)

    def test_mask_single_color(self):
        inp = [[1,2,3],[1,2,3],[1,2,3]]
        out = [[0,2,0],[0,2,0],[0,2,0]]
        s = solve(inp, out)
        assert s is not None and verify(s, inp, out)

    def test_flood_background(self):
        inp = [[0,1,0],[1,0,1],[0,1,0]]
        out = [[3,1,3],[1,3,1],[3,1,3]]
        s = solve(inp, out)
        assert s is not None and verify(s, inp, out)

    def test_invert_colors(self):
        inp = [[0,9],[1,8]]
        out = [[9,0],[8,1]]
        s = solve(inp, out)
        assert s is not None and verify(s, inp, out)

    def test_swap_two_colors(self):
        inp = [[0,1],[1,0]]
        out = [[1,0],[0,1]]
        s = solve(inp, out)
        assert s is not None and verify(s, inp, out)


# ─────────────────────────────────────────────────────────────────────────────
# Scale
# ─────────────────────────────────────────────────────────────────────────────

class TestScale:
    def test_scale_up_2x(self):
        inp = [[1,2],[3,4]]
        out = [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]
        s = solve(inp, out)
        assert s is not None and verify(s, inp, out)

    def test_tile_2x2(self):
        inp = [[1,0],[0,1]]
        out = [[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]]
        s = solve(inp, out)
        assert s is not None and verify(s, inp, out)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary / Robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundary:
    def test_impossible_returns_none(self):
        rng = np.random.default_rng(999)
        inp = rng.integers(0, 10, (4, 4))
        out = rng.integers(0, 10, (4, 4))
        s = run_evolution_engine(inp, out, max_expand=300, evolve_steps=5)
        assert s is None

    def test_shape_mismatch_no_crash(self):
        inp = np.array([[1,2],[3,4]])
        out = np.array([[1,2,3],[4,5,6],[7,8,9]])
        try:
            run_evolution_engine(inp, out, max_expand=100, evolve_steps=5)
        except Exception as e:
            pytest.fail(f"Crashed on shape mismatch: {e}")

    def test_1x1_trivial(self):
        assert solve([[5]], [[5]]) == []

    def test_all_zeros(self):
        assert solve([[0,0],[0,0]], [[0,0],[0,0]]) == []

    def test_large_grid_no_crash(self):
        rng = np.random.default_rng(1)
        inp = rng.integers(0, 3, (10, 10))
        out = np.fliplr(inp.copy())
        s = run_evolution_engine(inp, out, max_expand=5000, seed=1)
        assert s is None or verify(s, inp, out)


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeBase
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeBase:
    def test_default_weight_half(self, kb):
        assert kb.weight("nonexistent") == pytest.approx(0.5)

    def test_success_raises_weight(self, kb):
        w0 = kb.weight("flip_h")
        kb.record("flip_h", True)
        assert kb.weight("flip_h") > w0

    def test_failure_lowers_weight(self, kb):
        w0 = kb.weight("flip_h")
        kb.record("flip_h", False)
        assert kb.weight("flip_h") < w0

    def test_singleton(self):
        assert KnowledgeBase() is KnowledgeBase()

    def test_save_load(self, tmp_path, kb):
        kb.record("rot90", True)
        kb.record("rot90", True)
        p = tmp_path / "kb.json"
        kb.save(p)
        kb.reset()
        assert kb.weight("rot90") == pytest.approx(0.5)
        kb.load(p)
        assert kb.weight("rot90") > 0.5

    def test_top_k_ordering(self, kb):
        for _ in range(5):
            kb.record("flip_h", True)
        kb.record("rot90", False)
        top = kb.top_k(k=2)
        assert top[0][0] == "flip_h"

    def test_persistence_across_solvers(self, registry, kb):
        run_evolution_engine(
            np.array([[1,0],[0,0]]),
            np.array([[0,0],[1,0]]),
            seed=0,
        )
        # stats must be non-empty after solve
        assert len(kb.top_k(k=100)) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Performance Smoke
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformance:
    TIMEOUT = 3.0

    def test_geometric_task_fast(self):
        t0 = time.time()
        solve([[1,2,3],[4,5,6],[7,8,9]], [[3,2,1],[6,5,4],[9,8,7]])
        assert time.time() - t0 < self.TIMEOUT

    def test_color_task_fast(self):
        t0 = time.time()
        solve([[1,0],[0,1]], [[2,0],[0,2]])
        assert time.time() - t0 < self.TIMEOUT
