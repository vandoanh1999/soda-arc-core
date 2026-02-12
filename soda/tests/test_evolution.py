# soda/tests/test_evolution.py
from __future__ import annotations
import unittest
import numpy as np
from typing import List, Callable, Dict
from soda.evolution import run_evolution_engine, SelfEvolvingSolver

Array = np.ndarray
Primitive = Callable[[Array], Array]


class DeterministicPrimitives:
    @staticmethod
    def library() -> Dict[str, Primitive]:
        return {
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


class EvolutionEngineContract(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.primitives = DeterministicPrimitives.library()
        self.solver = SelfEvolvingSolver(self.primitives)

    def _apply_path(self, start: Array, path: List[str]) -> Array:
        grid = start.copy()
        for step in path:
            grid = self.primitives.get(step, self.solver.genome.library()[step])(grid)
        return grid

    def test_rotation_solution(self):
        start = np.array([[1, 2], [3, 4]])
        target = np.rot90(start, 1)
        path = self.solver.solve(start, target, max_depth=4, max_expand=5000, evolve_steps=10)
        self.assertIsNotNone(path)
        result = self._apply_path(start, path)
        self.assertTrue(np.array_equal(result, target))

    def test_flip_solution(self):
        start = np.array([[1, 0, 2], [3, 4, 5]])
        target = np.fliplr(start)
        path = self.solver.solve(start, target, max_depth=4, max_expand=5000, evolve_steps=10)
        self.assertIsNotNone(path)
        result = self._apply_path(start, path)
        self.assertTrue(np.array_equal(result, target))

    def test_shift_solution(self):
        start = np.array([[1, 2, 3], [4, 5, 6]])
        target = np.roll(start, 1, axis=1)
        path = self.solver.solve(start, target, max_depth=5, max_expand=8000, evolve_steps=20)
        self.assertIsNotNone(path)
        result = self._apply_path(start, path)
        self.assertTrue(np.array_equal(result, target))

    def test_no_solution_returns_none(self):
        start = np.array([[1, 2], [3, 4]])
        target = np.array([[9, 9], [9, 9]])
        path = self.solver.solve(start, target, max_depth=3, max_expand=2000, evolve_steps=5)
        self.assertIsNone(path)

    def test_genome_evolves(self):
        start = np.array([[1, 2], [3, 4]])
        target = np.rot90(start, 2)
        _ = self.solver.solve(start, target, max_depth=5, max_expand=8000, evolve_steps=50)
        evolved = len(self.solver.genome.genes) > 0
        self.assertTrue(evolved)

    def test_engine_entrypoint(self):
        start = np.array([[1, 2], [3, 4]])
        target = np.rot90(start, 3)
        path = run_evolution_engine(start, target)
        if path is not None:
            grid = start.copy()
            prims = DeterministicPrimitives.library()
            for p in path:
                grid = prims.get(p, prims.get(p, None) or prims.get(p, None) or grid)(grid)
            self.assertTrue(np.array_equal(grid, target))


if __name__ == "__main__":
    unittest.main()