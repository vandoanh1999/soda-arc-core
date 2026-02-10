# soda-arc-core

**SODA ARC Core** is a minimal, test-driven core for solving ARC-like grid reasoning tasks using
**frozen primitives + compositional reasoning**, without gradient-based learning.

This repository is intentionally small and explicit.
Every capability must be:
- Deterministic
- Testable
- Composable
- Reusable without retraining

## Philosophy

ARC tasks are not about pattern recognition at scale.
They are about:
- Discovering simple transformations
- Composing known operations
- Reusing frozen knowledge without catastrophic forgetting

SODA approaches ARC as:
> **Search over compositions of primitives on discrete grids**

No neural networks.  
No finetuning.  
No hidden state.

---

## Core Concepts

### Grid
A grid is a 2D integer matrix representing colors (0 = background).

### Primitive
A primitive is a **pure function**: