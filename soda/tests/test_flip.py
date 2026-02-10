from soda.grid import Grid
from soda.primitives import flip_horizontal

def test_flip():
    g = Grid([
        [1, 2],
        [3, 4],
    ])

    expected = Grid([
        [2, 1],
        [4, 3],
    ])

    assert flip_horizontal(g) == expected