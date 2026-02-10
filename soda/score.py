def score_solution(sequence, invariants):
    score = 0

    score += 10 - len(sequence)  # shorter = better

    if invariants.get("same_shape"):
        score += 2
    if invariants.get("color_preserved"):
        score += 2
    if invariants.get("color_subset"):
        score += 1

    return score