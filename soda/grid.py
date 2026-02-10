import numpy as np

class Grid:
    def __init__(self, data):
        self.data = np.array(data, dtype=int)

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return Grid(self.data.copy())

    def __eq__(self, other):
        return isinstance(other, Grid) and np.array_equal(self.data, other.data)

    def __repr__(self):
        return f"Grid(shape={self.data.shape})"