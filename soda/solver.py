from .search import search_composition
from .invariants import extract_invariants
from .score import score_solution

def solve_arc(input_grid, output_grid, anchors):
    invariants = extract_invariants(input_grid, output_grid)
    solutions = search_composition(
        input_grid,
        output_grid,
        anchors
    )

    ranked = []
    for seq in solutions:
        score = score_solution(seq, invariants)
        ranked.append((score, seq))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked