from soda.grid import Grid
from soda.primitives import translate

def test_translate_right():
    g = Grid([
        [1, 0],
        [0, 0],
    ])

    expected = Grid([
        [0, 1],
        [0, 0],
    ])

    assert translate(g, dx=1, dy=0) == expected