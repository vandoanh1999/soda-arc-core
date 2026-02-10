from __future__ import annotations
from typing import Callable, Tuple, Dict, Optional, List
import numpy as np
from soda.grid import Grid

PrimitiveFn = Callable[[Grid], Grid]


class Primitive:
    __slots__ = ("name", "fn", "_metadata")

    def __init__(self, name: str, fn: PrimitiveFn, metadata: Optional[Dict] = None):
        self.name = name
        self.fn = fn
        self._metadata = metadata or {}

    def __call__(self, grid: Grid) -> Grid:
        return self.fn(grid)

    def __repr__(self) -> str:
        return f"Primitive({self.name})"

    @property
    def metadata(self) -> Dict:
        return self._metadata.copy()


class PrimitiveFactory:
    @staticmethod
    def identity() -> Primitive:
        return Primitive("identity", lambda g: g.copy())

    @staticmethod
    def rotate(angle: int) -> Primitive:
        if angle not in (90, 180, 270, 360):
            raise ValueError("Angle must be 90, 180, 270, or 360")
        
        if angle == 360:
            return PrimitiveFactory.identity()
        
        k = angle // 90

        def _rotate(grid: Grid) -> Grid:
            return Grid(np.rot90(grid.data, k=-k))

        return Primitive(f"rotate_{angle}", _rotate, {"angle": angle, "category": "geometric"})

    @staticmethod
    def flip(axis: str) -> Primitive:
        axis_map = {
            "horizontal": (lambda g: Grid(np.flipud(g.data)), "flip_horizontal"),
            "vertical": (lambda g: Grid(np.fliplr(g.data)), "flip_vertical"),
            "diagonal": (lambda g: Grid(g.data.T), "flip_diagonal"),
            "anti_diagonal": (lambda g: Grid(np.rot90(g.data.T, 2)), "flip_anti_diagonal")
        }
        
        if axis not in axis_map:
            raise ValueError(f"Invalid axis: {axis}")
        
        fn, name = axis_map[axis]
        return Primitive(name, fn, {"axis": axis, "category": "geometric"})

    @staticmethod
    def translate(vector: Tuple[int, int], fill: int = 0, wrap: bool = False) -> Primitive:
        dr, dc = vector

        def _translate(grid: Grid) -> Grid:
            r, c = grid.shape
            
            if wrap:
                return Grid(np.roll(np.roll(grid.data, dr, axis=0), dc, axis=1))
            
            out = np.full((r, c), fill, dtype=np.int8)
            
            rs = max(0, dr)
            re = min(r, r + dr)
            cs = max(0, dc)
            ce = min(c, c + dc)
            
            src_rs = max(0, -dr)
            src_re = src_rs + (re - rs)
            src_cs = max(0, -dc)
            src_ce = src_cs + (ce - cs)
            
            out[rs:re, cs:ce] = grid.data[src_rs:src_re, src_cs:src_ce]
            return Grid(out)

        return Primitive(
            f"translate_{dr}_{dc}{'_wrap' if wrap else ''}",
            _translate,
            {"vector": vector, "fill": fill, "wrap": wrap, "category": "spatial"}
        )

    @staticmethod
    def color_map(mapping: Dict[int, int]) -> Primitive:
        def _map(grid: Grid) -> Grid:
            return grid.map_colors(mapping)
        return Primitive("color_map", _map, {"mapping": mapping, "category": "symbolic"})

    @staticmethod
    def scale(factor: int, sampling: str = "nearest") -> Primitive:
        if factor < 1:
            raise ValueError("Scale factor must be >= 1")

        def _scale(grid: Grid) -> Grid:
            if factor == 1:
                return grid.copy()
            
            r, c = grid.shape
            if sampling == "nearest":
                return Grid(np.repeat(np.repeat(grid.data, factor, axis=0), factor, axis=1))
            else:
                raise ValueError(f"Unsupported sampling: {sampling}")

        return Primitive(f"scale_{factor}", _scale, {"factor": factor, "category": "geometric"})

    @staticmethod
    def filter_color(keep_colors: set, fill: int = 0) -> Primitive:
        def _filter(grid: Grid) -> Grid:
            mask = np.isin(grid.data, list(keep_colors))
            result = np.where(mask, grid.data, fill)
            return Grid(result)
        
        return Primitive("filter_color", _filter, {"keep_colors": keep_colors, "category": "symbolic"})

    @staticmethod
    def crop_to_content(background: int = 0) -> Primitive:
        def _crop(grid: Grid) -> Grid:
            mask = grid.data != background
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not rows.any() or not cols.any():
                return Grid([[background]])
            
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            
            return grid.extract_region(r0, c0, r1 + 1, c1 + 1)
        
        return Primitive("crop_to_content", _crop, {"background": background, "category": "spatial"})

    @staticmethod
    def tile(rows: int, cols: int) -> Primitive:
        if rows < 1 or cols < 1:
            raise ValueError("Tile dimensions must be >= 1")

        def _tile(grid: Grid) -> Grid:
            return Grid(np.tile(grid.data, (rows, cols)))

        return Primitive(f"tile_{rows}x{cols}", _tile, {"rows": rows, "cols": cols, "category": "spatial"})

    @staticmethod
    def invert_colors(max_color: int = 9) -> Primitive:
        def _invert(grid: Grid) -> Grid:
            return Grid(max_color - grid.data)
        
        return Primitive("invert_colors", _invert, {"max_color": max_color, "category": "symbolic"})
