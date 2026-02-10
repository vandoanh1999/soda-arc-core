from __future__ import annotations
from typing import Tuple, Iterable, Dict, Optional, List, Set
import numpy as np

ArrayLike = Iterable[Iterable[int]]


class Grid:
    __slots__ = ("_data", "_shape", "_hash", "_color_histogram", "_symmetry_cache")

    def __init__(self, data: ArrayLike):
        arr = np.asarray(data, dtype=np.int8)
        if arr.ndim != 2:
            raise ValueError("Grid must be 2D")
        if arr.size == 0:
            raise ValueError("Grid cannot be empty")
        self._data = arr.copy()
        self._data.setflags(write=False)
        self._shape = self._data.shape
        self._hash: Optional[int] = None
        self._color_histogram: Optional[Dict[int, int]] = None
        self._symmetry_cache: Dict[str, bool] = {}

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def height(self) -> int:
        return self._shape[0]

    @property
    def width(self) -> int:
        return self._shape[1]

    def at(self, r: int, c: int) -> int:
        if not (0 <= r < self._shape[0] and 0 <= c < self._shape[1]):
            raise IndexError(f"Index ({r}, {c}) out of bounds for shape {self._shape}")
        return int(self._data[r, c])

    def safe_at(self, r: int, c: int, default: int = 0) -> int:
        if 0 <= r < self._shape[0] and 0 <= c < self._shape[1]:
            return int(self._data[r, c])
        return default

    def equals(self, other: Grid) -> bool:
        return self._shape == other._shape and np.array_equal(self._data, other._data)

    def copy(self) -> Grid:
        return Grid(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return False
        return self.equals(other)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._shape, self._data.tobytes()))
        return self._hash

    def __repr__(self) -> str:
        return f"Grid(shape={self._shape})"

    def color_histogram(self) -> Dict[int, int]:
        if self._color_histogram is None:
            unique, counts = np.unique(self._data, return_counts=True)
            self._color_histogram = dict(zip(unique.tolist(), counts.tolist()))
        return self._color_histogram.copy()

    def unique_colors(self) -> Set[int]:
        return set(self.color_histogram().keys())

    def most_common_color(self, exclude: Optional[Set[int]] = None) -> Optional[int]:
        hist = self.color_histogram()
        if exclude:
            hist = {k: v for k, v in hist.items() if k not in exclude}
        if not hist:
            return None
        return max(hist.items(), key=lambda x: x[1])[0]

    def map_colors(self, mapping: Dict[int, int]) -> Grid:
        vect = np.vectorize(lambda x: mapping.get(int(x), int(x)), otypes=[np.int8])
        return Grid(vect(self._data))

    def replace_color(self, old: int, new: int) -> Grid:
        return self.map_colors({old: new})

    def mask(self, color: int) -> np.ndarray:
        return self._data == color

    def extract_region(self, r0: int, c0: int, r1: int, c1: int) -> Grid:
        r0, r1 = max(0, r0), min(self._shape[0], r1)
        c0, c1 = max(0, c0), min(self._shape[1], c1)
        if r0 >= r1 or c0 >= c1:
            raise ValueError("Invalid region bounds")
        return Grid(self._data[r0:r1, c0:c1])

    def overlay(self, other: Grid, r_offset: int, c_offset: int, transparent: Optional[int] = None) -> Grid:
        result = self._data.copy()
        result.setflags(write=True)
        
        r0, c0 = max(0, r_offset), max(0, c_offset)
        r1 = min(self._shape[0], r_offset + other._shape[0])
        c1 = min(self._shape[1], c_offset + other._shape[1])
        
        src_r0 = max(0, -r_offset)
        src_c0 = max(0, -c_offset)
        src_r1 = src_r0 + (r1 - r0)
        src_c1 = src_c0 + (c1 - c0)
        
        patch = other._data[src_r0:src_r1, src_c0:src_c1]
        
        if transparent is not None:
            mask = patch != transparent
            result[r0:r1, c0:c1][mask] = patch[mask]
        else:
            result[r0:r1, c0:c1] = patch
        
        return Grid(result)

    def has_vertical_symmetry(self) -> bool:
        if "vertical" not in self._symmetry_cache:
            self._symmetry_cache["vertical"] = np.array_equal(self._data, np.fliplr(self._data))
        return self._symmetry_cache["vertical"]

    def has_horizontal_symmetry(self) -> bool:
        if "horizontal" not in self._symmetry_cache:
            self._symmetry_cache["horizontal"] = np.array_equal(self._data, np.flipud(self._data))
        return self._symmetry_cache["horizontal"]

    def has_diagonal_symmetry(self) -> bool:
        if "diagonal" not in self._symmetry_cache:
            if self._shape[0] == self._shape[1]:
                self._symmetry_cache["diagonal"] = np.array_equal(self._data, self._data.T)
            else:
                self._symmetry_cache["diagonal"] = False
        return self._symmetry_cache["diagonal"]

    def has_rotational_symmetry(self, order: int = 2) -> bool:
        key = f"rotational_{order}"
        if key not in self._symmetry_cache:
            if self._shape[0] != self._shape[1]:
                self._symmetry_cache[key] = False
            else:
                k = 4 // order
                self._symmetry_cache[key] = np.array_equal(self._data, np.rot90(self._data, k))
        return self._symmetry_cache[key]
