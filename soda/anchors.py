class Anchor:
    def __init__(self, name, primitive):
        self.name = name
        self.primitive = primitive
        self.frozen = True

    def apply(self, grid):
        return self.primitive(grid)

    def __repr__(self):
        return f"Anchor({self.name})"