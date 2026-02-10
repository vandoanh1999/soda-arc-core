from itertools import product

def search_composition(
    input_grid,
    target_grid,
    anchors,
    max_depth=3
):
    solutions = []

    for depth in range(1, max_depth + 1):
        for seq in product(anchors, repeat=depth):
            g = input_grid
            valid = True

            for anchor in seq:
                g = anchor.apply(g)
                if g.shape != target_grid.shape:
                    valid = False
                    break

            if valid and g == target_grid:
                solutions.append(seq)

    return solutions