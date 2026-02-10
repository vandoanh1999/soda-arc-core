from soda.grid import Grid
from soda.primitives import rotate_90

def test_rotate_3x3():
    g = Grid([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    expected = Grid([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ])

    assert rotate_90(g) == expected