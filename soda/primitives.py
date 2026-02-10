import numpy as np
from .grid import Grid

def rotate_90(grid: Grid) -> Grid:
    return Grid(np.rot90(grid.data, k=-1))

def flip_horizontal(grid: Grid) -> Grid:
    return Grid(np.fliplr(grid.data))

def translate(grid: Grid, dx: int, dy: int, fill=0) -> Grid:
    h, w = grid.shape
    out = np.full((h, w), fill, dtype=int)

    for y in range(h):
        for x in range(w):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                out[ny, nx] = grid.data[y, x]

    return Grid(out)