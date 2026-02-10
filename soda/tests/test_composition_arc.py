from soda.grid import Grid
from soda.primitives import rotate_90, flip_horizontal
from soda.anchors import Anchor
from soda.solver import solve_arc

def test_rotate_then_flip():
    input_grid = Grid([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    output_grid = Grid([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    anchors = [
        Anchor("rotate", rotate_90),
        Anchor("flip", flip_horizontal)
    ]

    solutions = solve_arc(input_grid, output_grid, anchors)

    assert len(solutions) > 0