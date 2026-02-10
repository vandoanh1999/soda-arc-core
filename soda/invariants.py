import numpy as np

def color_set(grid):
    return set(np.unique(grid.data))

def shape(grid):
    return grid.shape

def extract_invariants(input_grid, output_grid):
    inv = {}
    inv["same_shape"] = shape(input_grid) == shape(output_grid)
    inv["color_preserved"] = color_set(input_grid) == color_set(output_grid)
    inv["color_subset"] = color_set(output_grid).issubset(color_set(input_grid))
    return inv