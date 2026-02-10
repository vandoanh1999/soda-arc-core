from __future__ import annotations
from typing import Iterable, List, Callable, Optional
from soda.grid import Grid
from soda.anchors import Anchor, AnchorChain
import numpy as np


def apply_sequence(grid: Grid, anchors: Iterable[Anchor]) -> Grid:
    current = grid
    for anchor in anchors:
        current = anchor.apply(current)
    return current


def apply_parallel(grid: Grid, anchors: Iterable[Anchor], combiner: Callable[[List[Grid]], Grid]) -> Grid:
    results = [anchor.apply(grid) for anchor in anchors]
    return combiner(results)


def apply_conditional(grid: Grid, predicate: Callable[[Grid], bool], 
                     true_anchor: Anchor, false_anchor: Optional[Anchor] = None) -> Grid:
    if predicate(grid):
        return true_anchor.apply(grid)
    elif false_anchor is not None:
        return false_anchor.apply(grid)
    return grid.copy()


def apply_iterative(grid: Grid, anchor: Anchor, max_iterations: int = 100, 
                   convergence: Optional[Callable[[Grid, Grid], bool]] = None) -> Grid:
    current = grid
    for i in range(max_iterations):
        next_grid = anchor.apply(current)
        
        if convergence and convergence(current, next_grid):
            return next_grid
        
        if current.equals(next_grid):
            return next_grid
        
        current = next_grid
    
    return current


def apply_fork(grid: Grid, anchors: List[Anchor]) -> List[Grid]:
    return [anchor.apply(grid) for anchor in anchors]


def curry(anchor: Anchor) -> Callable[[Grid], Grid]:
    return lambda grid: anchor.apply(grid)


def uncurry(fn: Callable[[Grid], Grid], name: str = "uncurried") -> Anchor:
    from soda.primitives import Primitive
    return Anchor(Primitive(name, fn))


def compose_all(anchors: List[Anchor]) -> Anchor:
    if not anchors:
        raise ValueError("Cannot compose empty anchor list")
    
    result = anchors[0]
    for anchor in anchors[1:]:
        result = result.compose(anchor)
    return result


def map_grid_parallel(grids: List[Grid], anchor: Anchor) -> List[Grid]:
    return [anchor.apply(grid) for grid in grids]


def fold_grids(grids: List[Grid], combiner: Callable[[Grid, Grid], Grid]) -> Grid:
    if not grids:
        raise ValueError("Cannot fold empty grid list")
    
    result = grids[0]
    for grid in grids[1:]:
        result = combiner(result, grid)
    return result


class Pipeline:
    __slots__ = ("_stages", "_name")

    def __init__(self, name: str = "pipeline"):
        self._stages: List[Callable[[Grid], Grid]] = []
        self._name = name

    def add_anchor(self, anchor: Anchor) -> Pipeline:
        self._stages.append(anchor.apply)
        return self

    def add_function(self, fn: Callable[[Grid], Grid]) -> Pipeline:
        self._stages.append(fn)
        return self

    def execute(self, grid: Grid) -> Grid:
        current = grid
        for stage in self._stages:
            current = stage(current)
        return current

    def __call__(self, grid: Grid) -> Grid:
        return self.execute(grid)

    def __repr__(self) -> str:
        return f"Pipeline({self._name}, stages={len(self._stages)})"


def memoize(anchor: Anchor) -> Anchor:
    cache: dict = {}
    
    def _memoized(grid: Grid) -> Grid:
        key = hash(grid)
        if key not in cache:
            cache[key] = anchor.apply(grid)
        return cache[key]
    
    from soda.primitives import Primitive
    return Anchor(Primitive(f"memoized_{anchor.name}", _memoized))
