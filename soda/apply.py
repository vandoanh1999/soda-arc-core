from .grid import Grid

def apply_primitive(grid: Grid, primitive, *args, **kwargs) -> Grid:
    return primitive(grid, *args, **kwargs)